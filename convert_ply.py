import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import os
import argparse
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_heatmap_ply_from_npy(ply_path, npy_path, output_path, target_attr_name):
    """
    표준 PLY 파일과 NPY 데이터를 결합하여 히트맵 PLY 파일을 생성합니다.
    PLY 파일의 SH DC(f_dc) 속성을 NPY 데이터로 생성한 히트맵 색상으로 덮어씁니다.
    """
    logger.info(f"===== Starting PLY Conversion for {target_attr_name} =====")
    logger.info(f"Input PLY: {ply_path}")
    logger.info(f"Input NPY: {npy_path}")

    # 1. PLY 파일 로드 및 기본 검증
    try:
        plydata = PlyData.read(ply_path)
        vertex_data = plydata['vertex'].data
        num_points = len(vertex_data)
        logger.info(f"Loaded PLY file. Total points: {num_points}")
    except Exception as e:
        logger.error(f"❌ Error reading PLY file: {e}")
        return

    # 2. NPY 파일 로드 및 데이터 검증
    try:
        values = np.load(npy_path).flatten()
        logger.info(f"Loaded NPY file. Total values: {len(values)}")
    except Exception as e:
        logger.error(f"❌ Error loading NPY file: {e}")
        return

    if len(values) != num_points:
        logger.error(f"❌ Data length mismatch! PLY points ({num_points}) != NPY values ({len(values)}). Aborting.")
        return

    # 3. 히트맵 색상 생성
    logger.info("Generating heatmap colors from scalar data...")
    
    # 아웃라이어 제거 및 정규화 (Min-Max)
    v_min = np.percentile(values, 1) # 하위 1%
    v_max = np.percentile(values, 99) # 상위 99%
    
    logger.info(f"   Original data range: min={values.min():.4f}, max={values.max():.4f}")
    logger.info(f"   Normalization range (1%-99%): min={v_min:.4f}, max={v_max:.4f}")
    
    values_clipped = np.clip(values, v_min, v_max)
    
    if v_max - v_min < 1e-8:
        normalized_values = np.zeros_like(values_clipped)
        logger.warning("   (Warning: Data range is too small, setting colors to uniform blue.)")
    else:
        normalized_values = (values_clipped - v_min) / (v_max - v_min)

    # Colormap 적용 (Turbo)
    cmap = plt.get_cmap('turbo')
    rgb_colors = cmap(normalized_values)[:, :3]  # [N, 3], range 0~1

    # 4. RGB -> SH (DC) 변환
    # 뷰어에서 색상이 올바르게 보이도록 SH 0차 계수로 변환
    SH_C0 = 0.28209479177
    f_dc_heatmap = (rgb_colors - 0.5) / SH_C0
    logger.info("   Converted RGB heatmap to SH DC coefficients.")
    
    # 5. 기존 PLY 속성을 복사하고 f_dc만 교체
    new_data = []
    
    # 원본 PLY의 속성 순서와 dtype을 그대로 사용
    dtype_list = [p.name for p in plydata['vertex'].properties]
    new_elements = np.empty(num_points, dtype=plydata['vertex'].data.dtype)
    
    for idx, prop_name in enumerate(dtype_list):
        if prop_name.startswith('f_dc_'):
            # f_dc 속성만 히트맵 데이터로 덮어쓰기
            color_channel = int(prop_name.split('_')[-1])
            new_elements[prop_name] = f_dc_heatmap[:, color_channel]
            logger.debug(f"   Replacing {prop_name} with heatmap data.")
        else:
            # 나머지 속성(xyz, opacity, scale, rot, f_rest)은 원본 데이터 그대로 복사
            new_elements[prop_name] = vertex_data[prop_name]

    logger.info("Created new PLY element data with heatmap colors.")

    # 6. 저장
    el = PlyElement.describe(new_elements, 'vertex')
    
    # 기존 파일명에 접미사 추가
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    PlyData([el], text=False).write(output_path) 
    logger.info(f"✅ Success! Saved viewable heatmap PLY: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert standard 3DGS PLY and separate NPY data into a viewable heatmap PLY.")
    parser.add_argument("--ply_input", type=str, required=True, 
                        help="Path to the standard PLY file (e.g., point_cloud.ply).")
    parser.add_argument("--npy_input", type=str, required=True, 
                        help="Path to the corresponding NPY file (e.g., s_k.npy or E_k.npy).")
    parser.add_argument("--attr_name", type=str, required=True, 
                        help="Name of the attribute (e.g., s_k or E_k) for output file naming.")
    
    # 디버그 모드 추가
    parser.add_argument("--debug", action='store_true', help="Enable detailed debug logs.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # 출력 파일명 자동 생성
    input_dir = os.path.dirname(args.ply_input)
    ply_fname = os.path.splitext(os.path.basename(args.ply_input))[0]
    output_path = os.path.join(input_dir, f"{ply_fname}_heatmap_{args.attr_name}.ply")

    create_heatmap_ply_from_npy(args.ply_input, args.npy_input, output_path, args.attr_name)