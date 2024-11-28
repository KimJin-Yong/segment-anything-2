import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
model_cfg = "sam2_hiera_s.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

# FINE_TUNED_MODEL_WEIGHTS = r"C:\Users\dockn\SAM-code\segment-anything-2\notebooks\our_data\fine_tuned_sam2_1500.torch"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2)
# predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

# 사이드바에 이미지 업로드
st.sidebar.header("Upload an Image")
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 이미지 업로드 확인
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # SAM 모델에 이미지 설정
    predictor.set_image(image_np)

    # Streamlit 캔버스 설정
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # 채우기 색상
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=image,
        update_streamlit=True,
        width=image.width,
        height=image.height,
        drawing_mode="point",  # 포인트 클릭 모드로 설정
        key="canvas",
    )

    # 마우스 클릭으로 생성된 좌표 확인 및 세그멘테이션 수행
    if canvas_result.json_data is not None:
        points = []
        labels = []

        # 캔버스에서 클릭한 좌표 추출
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "circle":
                x, y = obj["left"], obj["top"]
                points.append([x, y])
                labels.append(1)  # Foreground 포인트로 설정

        if points:
            # SAM의 세그멘테이션 예측
            input_points = np.array(points)
            input_labels = np.array(labels)
            
            masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )

            # 마스크 이미지 생성
            mask_image = (masks[0] * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_image)

            # 원본 이미지와 마스크 겹쳐서 오버레이 생성
            overlay_image = image.copy()
            overlay_image.paste(ImageOps.colorize(mask_pil, black="black", white="blue").convert("RGBA"), (0, 0), mask_pil)

            # 결과 이미지들 표시
            st.image(mask_pil, caption="Segmentation Mask")
            st.image(overlay_image, caption="Original Image with Segmentation Overlay")
else:
    st.write("Please upload an image to start.")