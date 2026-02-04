# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
from typing import Optional

# 전개도 생성 모듈
from services.dieline_generate import DielineAnalyzer

# 배너 생성 모듈
from services.banner_generate import AdBannerGenerator

# 영상 생성 모듈
from services.video_generate import generate_video_for_product

# ✅ SNS 이미지 생성 모듈 (새로 추가)
from services.sns_image_generate import SNSImageGenerator

app = FastAPI(title="AI Product Media Server")


# ✅ sns / sns_background 추가
ALLOWED = {"package", "video", "poster", "dieline", "banner", "sns", "sns_background"}


# ===============================================================================================================================================================================================================
#  경로 체크
# ===============================================================================================================================================================================================================
def ensure_product_dir(product_id: int) -> Path:
    product_dir = str(product_id)
    product_dir.mkdir(parents=True, exist_ok=True)
    return product_dir


# ===============================================================================================================================================================================================================
# 1) 이미지 조회 API
# ===============================================================================================================================================================================================================
@app.get("/ai/{product_id}/images/{img_type}")
def get_img(product_id: int, img_type: str):
    if img_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="invalid type")

    # 이미지 타입별 파일명 매핑
    filename_map = {
        "package": "package.png",
        "poster": "poster.png",
        "dieline": "dieline.png",
        "banner": "banner.png",
        # ✅ sns 결과
        "sns": "sns.png",
        "sns_background": "sns_background.png",
    }

    if img_type == "video":
        # 기존 코드가 이미지 전용이므로 video는 제외하거나 별도 endpoint 사용 권장
        raise HTTPException(status_code=400, detail="video is not an image type")

    filename = filename_map.get(img_type, f"{img_type}.png")
    path = str(product_id) / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")

    return FileResponse(path, media_type="image/png")


# ===============================================================================================================================================================================================================
# 2) 더미 패키지 생성 API
# ===============================================================================================================================================================================================================
@app.post("/ai/{product_id}/package")
async def create_package_dummy(product_id: int, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image upload allowed")
    product_dir = ensure_product_dir(product_id)
    out_path = product_dir / "package.png"
    with out_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return FileResponse(out_path, media_type="image/png")


# ===============================================================================================================================================================================================================
# 3) 배너 생성 API
# ===============================================================================================================================================================================================================
class BannerRequest(BaseModel):
    headline: str = Field(..., example="맛있는 간식")
    typo_text: str = Field(..., example="손이가요 손이가.")


@app.post("/ai/{product_id}/banner")
def create_banner_from_file(
    product_id: int,
    headline: str = Form(...),
    typo_text: str = Form(...),
):
    product_dir = ensure_product_dir(product_id)
    input_path = product_dir / "package.png"  # 디렉터리에서 가져오는 이미지
    output_path = product_dir / "banner.png"

    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"{input_path} not found")

    try:
        generator = AdBannerGenerator(api_key="김채환")
        generator.process(
            image_path=str(input_path),
            headline=headline,
            typo_text=typo_text,
            output_path=str(output_path),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Banner generation failed: {e}")

    return FileResponse(output_path, media_type="image/png", filename="banner.png")


# ===============================================================================================================================================================================================================
# 4) 영상 생성 API
# ===============================================================================================================================================================================================================
class VideoGenRequest(BaseModel):
    food_name: str = Field(..., example="새우깡")
    food_type: str = Field(..., example="스낵")
    ad_concept: str = Field(..., example="감성+트렌디")
    ad_req: str = Field(
        ..., example="바삭함, 중독성, 가벼운 간식, 자연스러운 일상 무드"
    )


@app.post("/ai/{product_id}/video")
async def create_video(
    product_id: int, req: VideoGenRequest, file: UploadFile = File(...)
):
    product_dir = ensure_product_dir(product_id)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image upload allowed")
    package_path = product_dir / "package.png"
    with package_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        final_mp4_path = await generate_video_for_product(
            product_id=product_id, req=req, product_image=file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video generation failed: {e}")

    return FileResponse(
        final_mp4_path, media_type="video/mp4", filename="final_video.mp4"
    )


# ===============================================================================================================================================================================================================
# 5) 전개도(Dieline) 분석 API
# ===============================================================================================================================================================================================================
@app.post("/ai/{product_id}/dieline")
def analyze_dieline(product_id: int, file: UploadFile = File(...)):
    # 1. 파일 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    # 2. 경로 설정 및 이미지 저장
    product_dir = ensure_product_dir(product_id)
    input_path = product_dir / "dieline_input.png"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. 분석 모듈 실행
    analyzer = DielineAnalyzer()
    try:
        result = analyzer.analyze(image_path=str(input_path), output_dir=product_dir)
        result["result_image_url"] = f"/ai/{product_id}/images/dieline"
        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ===============================================================================================================================================================================================================
# 6) ✅ SNS 인스타 광고 이미지 생성 API (새 기능)
# ===============================================================================================================================================================================================================
class SNSGenRequest(BaseModel):
    main_text: str = Field(..., example="나야 새우깡")
    sub_text: str = Field("", example="바삭함의 정석")
    preset: Optional[str] = Field(None, example="ocean_sunset")
    custom_prompt: Optional[str] = Field(
        None, example="A dramatic night beach scene with crashing waves..."
    )
    save_background: bool = Field(True, example=True)


@app.post("/ai/{product_id}/sns")
def create_sns_image(product_id: int, req: SNSGenRequest):
    """
    입력:
      - main_text, sub_text
      - preset 또는 custom_prompt
    출력:
      - sns_background.png (옵션)
      - sns.png (최종 1080x1350)
    """
    product_dir = ensure_product_dir(product_id)

    # 상품 이미지는 기존처럼 package.png를 사용
    product_path = product_dir / "package.png"
    if not product_path.exists():
        raise HTTPException(
            status_code=404, detail="package.png not found. Upload package first."
        )

    background_path = product_dir / "sns_background.png"
    final_path = product_dir / "sns.png"

    try:
        generator = SNSImageGenerator()  # 내부에서 GEMINI_API_KEY 사용
        generator.generate(
            product_path=str(product_path),
            main_text=req.main_text,
            sub_text=req.sub_text or "",
            preset=req.preset,
            custom_prompt=req.custom_prompt,
            output_path=str(final_path),
            save_background=req.save_background,
            background_output_path=(
                str(background_path) if req.save_background else None
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SNS generation failed: {e}")

    return {
        "product_id": product_id,
        "sns_image_url": f"/ai/{product_id}/images/sns",
        "background_image_url": (
            f"/ai/{product_id}/images/sns_background"
            if req.save_background
            else None
        ),
        "output_path": str(final_path),
    }
