from . import airsim
import gym
import numpy as np
import cv2


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        self.image_shape = image_shape
        self.sections = env_config["sections"]

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(9)

        self.info = {"collision": False}

        self.collision_time = 0
        self.random_start = True
        self.last_hole_reached = False
        self.setup_flight()

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()

        pose = self.drone.simGetVehiclePose()
        x = pose.position.x_val
        y = pose.position.y_val
        z = pose.position.z_val

        # Wall/hole parameters:
        # The original code assuems the wall is 3.7m in front of the start x position.
        wall_x = self.agent_start_pos + 3.7
        hole_y, hole_z = self.target_pos  # setup_flight에서 sections[...]에서 가져온 y,z
        hole_radius = 0.35  # Slight tolerance; can lower to 0.3

        info["hole_reached"] = False

        # Determine if the drone passed through the hole when crossing the wall x-position
        if not done and x >= wall_x:
            # Compute distance in the YZ-plane (consistent with compute_reward)
            dist_yz = np.linalg.norm(np.array([y, -z]) - np.array(self.target_pos))

            done = 1  # Episode ends once drone reaches the wall
            if dist_yz <= hole_radius:
                reward += 100.0
                info["hole_reached"] = True
            else:
                reward -= 100.0
                info["hole_reached"] = False
            
        self.last_hole_reached = info["hole_reached"]
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        self.drone.moveToZAsync(-1, 1)

        # Record initial collision timestamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        # Randomly select a section
        if self.random_start == True:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_pos = section["offset"][0]
        self.target_pos = section["target"]

        # Start drone at a random (y, z) position
        y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        # Compute initial distance to target (used for reward shaping)
        self.target_dist_prev = np.linalg.norm(np.array([y_pos, z_pos]) - self.target_pos)

    def do_action(self, select_action):
        speed = 0.4

        # Mapping action index to velocity commands
        if select_action == 0:
            vy, vz = (-speed, -speed)
        elif select_action == 1:
            vy, vz = (0, -speed)
        elif select_action == 2:
            vy, vz = (speed, -speed)
        elif select_action == 3:
            vy, vz = (-speed, 0)
        elif select_action == 4:
            vy, vz = (0, 0)
        elif select_action == 5:
            vy, vz = (speed, 0)
        elif select_action == 6:
            vy, vz = (-speed, speed)
        elif select_action == 7:
            vy, vz = (0, speed)
        else:
            vy, vz = (speed, speed)

        # Execute movement in body frame
        self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, duration=1).join()

        # Stabilize drone after movement
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0
        hole_reached = False

        # Distance-based shaping reward
        x,y,z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([y,-z]) - self.target_pos)
        reward += (self.target_dist_prev - target_dist_curr)*20
        
        self.target_dist_prev = target_dist_curr

        agent_traveled_x = np.abs(self.agent_start_pos - x)
        

        # Alignment reward when approaching target
        if target_dist_curr < 0.30:
            reward += 12.0
            # Alignment becomes more important when agent is close to the hole 
            if agent_traveled_x > 2.9:
                reward += 7.0

        elif target_dist_curr < 0.45:
            reward += 7.0

        # Collision penalty
        if self.is_collision():
            reward = -100.0
            done = 1

        # Passed hole successfully
        elif agent_traveled_x > 3.7:
            reward += 10.0
            done = 1
            hole_reached = True

        # Check if the hole disappeared from camera frame
        # (target_dist_curr-0.3) : distance between agent and hole's end point
        # (3.7-agent_traveled_x) : distance between agent and wall
        # (3.7-agent_traveled_x)*sin(60) : end points that camera can capture
        # FOV : 120 deg, sin(60) ~ 1.732 
        elif (target_dist_curr-0.3) > (3.7-agent_traveled_x)*1.732:
            reward = -100.0
            done = 1

        self.last_hole_reached = hole_reached
        # hole_reached = getattr(self, "last_hole_reached", False)
        # print("> Hole reached (last):", hole_reached)

        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])

        if not responses or responses[0].height == 0 or responses[0].width == 0:
            # Sometimes API returns empty response
            return np.zeros(self.image_shape, dtype=np.uint8)
        
        # numpy 2.0이상에서는 fromstring대신 frombufffer 사용해야 함
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)

        # Sometimes no image returns from api
        try: 
            h = responses[0].height
            w = responses[0].width
            img2d = img1d.reshape(h, w, 3)  # The original size so far (ex: 192 x 192 x 3)

            # 우리 env가 기대하는 크기로 리사이즈
            target_h, target_w, target_c = self.image_shape
            img_resized = cv2.resize(img2d, (target_w, target_h))  # (w, h) 순서 주의
            return img_resized.reshape(self.image_shape)
        except Exception as e:
            print(f"[WARN] get_rgb_image reshape failed: {e}")
            return np.zeros(self.image_shape, dtype=np.uint8)

    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image


class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config):
        # Statistics for evaluation
        self.eps_n = 0                      # episode count 
        self.agent_traveled = []            # 각 에피소드에서 x 방향 이동 거리들
        self.hole_reached_last_k = []       # 최근 k개 에피소드에서 성공 여부 기록
        self.hole_reached_total = 0         # The number of passed hole so far 
        self.random_start = False           # Deterministic start section

        # 베이스 클래스 초기화 (여기서 AirSim 연결 + setup_flight 호출됨)
        super(TestEnv, self).__init__(ip_address, image_shape, env_config)

    def setup_flight(self):
        # Base setup: reset drone, arm, section seleciton, etc.
        super(TestEnv, self).setup_flight()
        self.eps_n += 1

        # Always start at (y=0, z=-2) for stable evaluation flight
        y_pos = 0.0
        z_pos = -2.0        # Height above ground (AirSim NED)

        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos, y_pos, z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
    def compute_reward(self):
        """
        Evalatuion reward:
        - Reward is irrelevant (policy already trained)
        - End of each episode:
          * Record forward distance 
          * Record pass/fail 
          * Print summary every 10 episodes
        """
        reward = 0
        done = 0

        # 현재 x 위치, 시작점으로부터 얼마나 전진했는지
        x, _, _ = self.drone.simGetVehiclePose().position
        agent_traveled_x = float(np.abs(self.agent_start_pos - x))

        # 이번 에피소드에서 hole 통과했는지 여부
        # (원래 코드 기준: x 방향으로 3.7m 넘게 가면 벽/구멍까지 도달했다고 가정)
        hole_reached_this_ep = False

        # Collision -> end episode
        if self.is_collision():
            done = 1

        # Reached wall/hole x_position -> treat as success
        elif agent_traveled_x > 3.7:
            hole_reached_this_ep = True
            reward += 1.0

        # End-of-episode statistics
        if done:
            # Record Moving distance  
            self.agent_traveled.append(agent_traveled_x)

            # Keeping track of success or failure (0 or 1) this episode
            self.hole_reached_last_k.append(int(hole_reached_this_ep))
            self.hole_reached_total += int(hole_reached_this_ep)

            # Maintaining the latest 10 
            if len(self.hole_reached_last_k) > 10:
                self.hole_reached_last_k.pop(0)

            # Printing out whether it has been successful or not per episode
            print("> Hole reached in this episode:", hole_reached_this_ep)

            # Summary of statistics for every 10 episode 
            if self.eps_n % 10 == 0:
                distances = np.array(self.agent_traveled, dtype=np.float32)
                last_k = min(10, len(self.hole_reached_last_k))
                holes_last_k = sum(self.hole_reached_last_k[-last_k:])

                print("---------------------------------")
                print("> Total episodes:", self.eps_n)
                print("> Flight distance (mean): %.2f" % (distances.mean()))
                print("> Holes reached (last %d): %d" % (last_k, int(holes_last_k)))
                print("> Holes reached (total): %d" % int(self.hole_reached_total))
                print("---------------------------------\n")

        return reward, done
