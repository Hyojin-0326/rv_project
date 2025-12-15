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

def create_heatmap_ply_from_npy(ply_path, npy_path, output_path, target_attr_name, rm_sk=False, threshold=0.01):
    """
    표준 PLY 파일과 NPY 데이터를 결합하여 히트맵 PLY 파일을 생성합니다.
    옵션에 따라 특정 임계값 이하의 포인트는 제거합니다.
    """
    logger.info(f"===== Starting PLY Conversion for {target_attr_name} =====")
    logger.info(f"Input PLY: {ply_path}")
    logger.info(f"Input NPY: {npy_path}")

    # 1. PLY 파일 로드
    try:
        plydata = PlyData.read(ply_path)
        vertex_data = plydata['vertex'].data
        num_points = len(vertex_data)
        logger.info(f"Loaded PLY file. Total points: {num_points}")
    except Exception as e:
        logger.error(f"❌ Error reading PLY file: {e}")
        return

    # 2. NPY 파일 로드
    try:
        values = np.load(npy_path).flatten()
        logger.info(f"Loaded NPY file. Total values: {len(values)}")
    except Exception as e:
        logger.error(f"❌ Error loading NPY file: {e}")
        return

    if len(values) != num_points:
        logger.error(f"❌ Data length mismatch! PLY points ({num_points}) != NPY values ({len(values)}). Aborting.")
        return

    # -----------------------------------------------------------
    # [NEW] 포인트 제거 로직 (rm_sk=True일 경우)
    # -----------------------------------------------------------
    if rm_sk:
        logger.info(f"🔍 Filtering points where {target_attr_name} < {threshold} ...")
        
        # 값이 threshold보다 큰 것만 남김 (절댓값 기준이 안전함)
        valid_mask = np.abs(values) >= threshold
        
        # 마스킹 적용
        vertex_data = vertex_data[valid_mask]
        values = values[valid_mask]
        
        new_num_points = len(values)
        removed_count = num_points - new_num_points
        
        logger.info(f"   Removed: {removed_count} points (Values close to 0)")
        logger.info(f"   Remaining: {new_num_points} points")
        
        if new_num_points == 0:
            logger.error("❌ All points were filtered out! Try lowering the threshold.")
            return
            
        num_points = new_num_points  # 포인트 개수 업데이트

    # 3. 히트맵 색상 생성
    logger.info("Generating heatmap colors from scalar data...")
    
    # 아웃라이어 제거 및 정규화 (Min-Max)
    v_min = np.percentile(values, 1) # 하위 1%
    v_max = np.percentile(values, 99) # 상위 99%
    
    logger.info(f"   Data range (after filter): min={values.min():.4f}, max={values.max():.4f}")
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
    SH_C0 = 0.28209479177
    f_dc_heatmap = (rgb_colors - 0.5) / SH_C0
    logger.info("   Converted RGB heatmap to SH DC coefficients.")
    
    # 5. 새로운 PLY 데이터 생성
    # 기존 vertex_data의 구조(dtype)를 유지하면서 필터링된 개수만큼 생성
    new_elements = np.empty(num_points, dtype=vertex_data.dtype)
    
    dtype_list = vertex_data.dtype.names
    
    for prop_name in dtype_list:
        if prop_name.startswith('f_dc_'):
            # f_dc 속성만 히트맵 데이터로 덮어쓰기
            color_channel = int(prop_name.split('_')[-1])
            new_elements[prop_name] = f_dc_heatmap[:, color_channel]
        else:
            # 나머지 속성(xyz, opacity, scale, rot 등)은 필터링된 vertex_data에서 복사
            new_elements[prop_name] = vertex_data[prop_name]

    logger.info("Created new PLY element data with heatmap colors.")

    # 6. 저장
    el = PlyElement.describe(new_elements, 'vertex')
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    PlyData([el], text=False).write(output_path) 
    logger.info(f"✅ Success! Saved viewable heatmap PLY: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert standard 3DGS PLY and separate NPY data into a viewable heatmap PLY.")
    parser.add_argument("--ply_input", type=str, required=True, 
                        help="Path to the standard PLY file.")
    parser.add_argument("--npy_input", type=str, required=True, 
                        help="Path to the corresponding NPY file (e.g., s_k.npy).")
    parser.add_argument("--attr_name", type=str, required=True, 
                        help="Name of the attribute (e.g., s_k) for output naming.")
    
    # [NEW] 추가된 옵션들
    parser.add_argument("--rm_sk", action='store_true', 
                        help="If set, removes points where the scalar value is close to 0.")
    parser.add_argument("--threshold", type=float, default=0.01, 
                        help="Threshold for removal. Points with value < threshold are removed. Default: 0.01")

    parser.add_argument("--debug", action='store_true', help="Enable detailed debug logs.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # 출력 파일명 생성 (옵션에 따라 이름 변경)
    input_dir = os.path.dirname(args.ply_input)
    ply_fname = os.path.splitext(os.path.basename(args.ply_input))[0]
    
    suffix = f"heatmap_{args.attr_name}"
    if args.rm_sk:
        suffix += "_pruned"
        
    output_path = os.path.join(input_dir, f"{ply_fname}_{suffix}.ply")

    create_heatmap_ply_from_npy(
        args.ply_input, 
        args.npy_input, 
        output_path, 
        args.attr_name,
        rm_sk=args.rm_sk,
        threshold=args.threshold
    )