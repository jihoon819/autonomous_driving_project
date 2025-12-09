import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Waymo Open Dataset의 프로토콜 버퍼 정의를 가져옵니다.
from waymo_open_dataset.protos import scenario_pb2

# 1. 다운로드 받은 tfrecord 파일 경로를 설정하세요.
FILENAME = 'YOUR_DOWNLOADED_FILE_NAME.tfrecord'  # <-- 파일 경로 수정 필요

def extract_and_plot_scenarios(filename, num_scenarios_to_check=5):
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    
    cnt = 0
    for data in dataset:
        # 2. 바이너리 데이터를 Scenario 프로토콜 버퍼로 파싱
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        
        # 3. AV(SDC: Self-Driving Car)의 트랙 인덱스 찾기
        sdc_track_index = scenario.sdc_track_index
        track = scenario.tracks[sdc_track_index]
        
        # 4. AV의 x, y 궤적 추출
        # states는 시간 순서대로 저장되어 있습니다.
        xs = []
        ys = []
        valid_states = [] # 데이터가 유효한지 체크
        
        for state in track.states:
            if state.valid: # 유효한 데이터만 추출
                xs.append(state.center_x)
                ys.append(state.center_y)
                valid_states.append(state)
                
        # 5. 시각화 (Matplotlib)
        # 직관적으로 단순한 경로인지 확인하기 위해 플롯을 그립니다.
        plt.figure(figsize=(6, 6))
        plt.plot(xs, ys, 'b-', linewidth=2, label='AV Trajectory')
        plt.scatter(xs[0], ys[0], c='g', label='Start') # 시작점
        plt.scatter(xs[-1], ys[-1], c='r', label='End') # 끝점
        
        plt.title(f"Scenario ID: {scenario.scenario_id}\nNum Agents: {len(scenario.tracks)}")
        plt.xlabel("Global X (m)")
        plt.ylabel("Global Y (m)")
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"[{cnt+1}] Scenario ID: {scenario.scenario_id}")
        print(f"    - Duration: {len(xs) * 0.1:.1f} sec") # 보통 10Hz 데이터
        print(f"    - Path Type Check: 눈으로 확인하세요 (직선/완만한 곡선 추천)")
        
        cnt += 1
        if cnt >= num_scenarios_to_check:
            break

# 실행
if __name__ == "__main__":
    # 파일 경로가 올바른지 확인 후 실행하세요.
    try:
        extract_and_plot_scenarios(FILENAME)
    except Exception as e:
        print(f"에러 발생: {e}")
        print("파일 경로가 맞는지, 라이브러리가 설치되었는지 확인해주세요.")