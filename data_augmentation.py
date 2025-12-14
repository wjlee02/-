import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def augment_data_darken(hdf5_path, new_hdf5_path, darken_factor, cameras):
    """
    주어진 hdf5_path의 데이터를 복사한 뒤,
    cameras 목록에 해당하는 이미지들의 밝기를 darken_factor만큼 조정하여 저장.
    """
    with h5py.File(hdf5_path, 'r') as f:
        with h5py.File(new_hdf5_path, 'w') as new_f:
            # 전체 그룹 구조 복사
            for key in f.keys():
                f.copy(key, new_f, name=key)

            for im_name in cameras:
                # 원본 이미지 로드
                dataset_path = f"observations/images/{im_name}"
                images = np.array(f[dataset_path])

                # 밝기 조정
                darker_images = np.clip(images * darken_factor, 0, 255).astype(np.uint8)

                # 기존 데이터셋 삭제 후 새로 생성
                del new_f[dataset_path]
                new_f.create_dataset(dataset_path, data=darker_images)

def augment_data_scale_and_shift(
    hdf5_path,
    new_hdf5_path,
    cameras,
    scale_range=(0.6, 1.0),
    fill_value=128
):
    """
    원본 hdf5 파일(hdf5_path)에서 데이터를 복사한 뒤,
    images/{camera} 데이터를 '축소 후 무작위 위치에 배치'하는 방식으로 변환하여
    new_hdf5_path에 저장합니다.
    
    1) scale_range 범위에서 무작위로 scale_factor를 뽑아 이미지를 축소
    2) 축소된 이미지를 (H,W) 크기의 새 캔버스에 무작위 오프셋(offset)으로 배치
    3) 남은 여백은 fill_value(기본 128)로 채움
    
    Parameters
    ----------
    hdf5_path : str
        원본 hdf5 파일 경로
    new_hdf5_path : str
        새로 생성할 hdf5 파일 경로
    cameras : list
        수정할 이미지 이름 목록 (예: ['camera1', 'camera2', 'camera3'])
    scale_range : tuple
        (min_scale, max_scale) 형태. 예: (0.8, 1.0)
    fill_value : int
        배경(여백)을 채울 값. 기본값=128 (회색)
    """
    min_scale, max_scale = scale_range
    
    # 1) 무작위 스케일링 비율 선택
    scale_factor = np.random.uniform(min_scale, max_scale)
    new_h = int(120 * scale_factor)
    new_w = int(160 * scale_factor)
    
    # 4) 새 캔버스 내에서 무작위 오프셋 위치 구하기
    #    (축소된 이미지가 완전히 들어갈 수 있는 범위 내)
    offset_y = np.random.randint(0, 120 - new_h + 1)
    offset_x = np.random.randint(0, 160 - new_w + 1)
    
    
    
    # 원본 HDF5 파일 열기
    with h5py.File(hdf5_path, 'r') as f:
        # 새로 저장할 HDF5 파일 열기
        with h5py.File(new_hdf5_path, 'w') as new_f:
            # 전체 그룹 구조 복사
            for key in f.keys():
                f.copy(key, new_f, name=key)

            # 지정된 camera들에 대해 축소&이동 수행
            for im_name in cameras:
                dataset_path = f"observations/images/{im_name}"
                images = np.array(f[dataset_path])  # (N, H, W, C) 형태

                N, H, W, C = images.shape
                transformed_images = np.zeros_like(images)

                for i in range(N):

                    # 2) 이미지를 (new_w, new_h)로 축소 (cv2 사용)
                    #   OpenCV의 resize 함수는 (width, height) 순서로 받습니다.
                    scaled_img = cv2.resize(
                        images[i],
                        (new_w, new_h),
                        interpolation=cv2.INTER_LINEAR
                    )

                    # 3) (H, W) 크기의 캔버스 생성 후 fill_value(128)로 초기화
                    canvas = np.full((H, W, C), fill_value, dtype=images.dtype)

                    

                    # 5) 축소된 이미지를 해당 오프셋 위치에 복사
                    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w, :] = scaled_img
                    
                    # 완성된 이미지를 저장
                    transformed_images[i] = canvas

                # 기존 데이터셋 삭제 후 새로 생성
                del new_f[dataset_path]
                new_f.create_dataset(dataset_path, data=transformed_images)

def augment_data_bgr(hdf5_path, new_hdf5_path, cameras):
    """
    BGR 채널 이미지를 RGB로 바꿔서 저장해주는 함수.
    """
    with h5py.File(hdf5_path, 'r') as f:
        with h5py.File(new_hdf5_path, 'w') as new_f:
            # 전체 그룹 구조 복사
            for key in f.keys():
                f.copy(key, new_f, name=key)

            for im_name in cameras:
                dataset_path = f"observations/images/{im_name}"
                images = np.array(f[dataset_path])

                # BGR -> RGB 변환(마지막 채널을 뒤집음)
                # 또는 images[..., [2,1,0]] 와 같이 명시적으로 채널 순서 바꿀 수도 있음
                rgb_images = images[..., ::-1]

                del new_f[dataset_path]
                new_f.create_dataset(dataset_path, data=rgb_images)

if __name__ == "__main__":
    # 기본 설정
    dir = "./datasets"
    # work = "box"
    data_dir = "box"
    
    # 예제: original 폴더에서 이미지를 불러와서
    #       밝기 조정, crop, bgr->rgb 등을 적용해서 aug 폴더에 저장
    folders = ['original']

    crop_num = 1

    d_count = 0
    for folder in folders:
        original_data_dir = f"{dir}/{data_dir}/{folder}"
        data_len = len(os.listdir(original_data_dir))

        for i in range(data_len):
            hdf5_path = f"{original_data_dir}/episode_{i}.hdf5"

            new_dir = f"{dir}/{data_dir}/aug"
            os.makedirs(new_dir, exist_ok=True)
            
            os.makedirs(f"{new_dir}/dark", exist_ok=True)
            os.makedirs(f"{new_dir}/light", exist_ok=True)
            os.makedirs(f"{new_dir}/crop", exist_ok=True)
            os.makedirs(f"{new_dir}/bgr", exist_ok=True)

            ####################################################
            # 1) 어두운(darken_factor=0.7) 버전
            ####################################################
            new_hdf5_path = f"{new_dir}/dark/episode_{d_count}.hdf5"
            augment_data_darken(hdf5_path, new_hdf5_path, 0.7, cameras=['camera1', 'camera2', 'camera'])
            print(f"{d_count}번(어두운) 에피소드가 저장되었습니다.")
            
            ####################################################
            # 2) 밝은(darken_factor=1.3) 버전
            ####################################################
            new_hdf5_path = f"{new_dir}/light/episode_{d_count}.hdf5"
            augment_data_darken(hdf5_path, new_hdf5_path, 1.3, cameras=['camera1', 'camera2', 'camera'])
            print(f"{d_count}번(밝은) 에피소드가 저장되었습니다.")
            
            ####################################################
            # 3) Cropping 버전 (예: x=50, y=50, w=200, h=200)
            ####################################################
            for i in range(crop_num):
                new_hdf5_path = f"{new_dir}/crop/episode_{d_count * crop_num + i}.hdf5"
                augment_data_scale_and_shift(hdf5_path, new_hdf5_path, cameras=['camera1', 'camera2', 'camera'])
                print(f"{d_count}번(크롭) 에피소드가 저장되었습니다.")


            ####################################################
            # 4) BGR -> RGB 변환 버전
            ####################################################
            new_hdf5_path = f"{new_dir}/bgr/episode_{d_count}.hdf5"
            augment_data_bgr(hdf5_path, new_hdf5_path, cameras=['camera1', 'camera2', 'camera'])
            print(f"{d_count}번(BGR->RGB) 에피소드가 저장되었습니다.")
            
            
            d_count += 1