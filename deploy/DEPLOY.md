# Deploy: Vercel (giao diện) + Modal (backend AI)

- **Modal** chạy backend FastAPI (TensorFlow + YOLO). Model nằm trong Modal Volume `cashew-models`,
  không nằm trong Git hay image. Container tự tắt khi không dùng.
- **Vercel** chỉ host giao diện tĩnh; trình duyệt gọi thẳng API Modal (CORS đã mở trong `server.py`).

Hiện tại:
- Backend: https://nguyenkimlong261105--cashew-leaf-api-web.us-east.modal.run
- Web: https://cashew-leaf-diagnosis.vercel.app

## Cài một lần

```bash
uv tool install modal          # hoặc: pip install modal
modal setup                    # đăng nhập qua trình duyệt
npm i -g vercel && vercel login
```

## Backend (Modal)

```bash
# Chỉ làm lại khi model đổi. Trên Git Bash đặt MSYS_NO_PATHCONV=1 để "/pipeline_bundle" không bị đổi thành đường dẫn Windows.
modal volume create cashew-models
modal volume put cashew-models EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/pipeline_bundle /pipeline_bundle

modal deploy deploy/modal_app.py      # sửa code Application/ thì chạy lại lệnh này
```

Kiểm tra: `<url>/api/health`, `<url>/api/model-info` (phải là EXP-035). Lần gọi đầu (cold start) mất khoảng 1,5 phút
vì nạp 5 DenseNet + YOLO; sau đó giữ ấm 5 phút.

## Giao diện (Vercel)

```bash
python deploy/prepare.py --api-url https://nguyenkimlong261105--cashew-leaf-api-web.us-east.modal.run
cd deploy/_build/vercel
vercel deploy --prod --yes --name cashew-leaf-diagnosis
```

Sửa giao diện (`Application/static`) → chạy lại hai lệnh trên.

## Lưu ý

- Gói Starter của Modal: chưa thêm thẻ chỉ có 1 USD; thêm thẻ để có 30 USD/tháng và đặt *spending limit* ở Settings → Billing.
- Endpoint backend công khai: ai biết URL cũng gọi được và tiêu tín dụng. Muốn giới hạn, đổi `allow_origins=["*"]`
  trong `Application/server.py` thành URL Vercel (không chặn được gọi trực tiếp bằng curl).
- Không commit model lên GitHub (`.gitignore` đã chặn `*.keras`, `*.pt`).
