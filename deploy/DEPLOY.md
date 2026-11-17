# Deploy: Vercel (giao diện) + Hugging Face Spaces (backend AI)

Vercel chỉ host giao diện tĩnh. Backend FastAPI (TensorFlow + YOLO + ~240 MB model) chạy trong Docker trên
Hugging Face Space (CPU miễn phí, 16 GB RAM). Trình duyệt gọi thẳng Space; CORS của `server.py` đã mở.

## 1. Tạo Space (backend)

1. Đăng nhập https://huggingface.co → **New Space** → SDK **Docker** → đặt tên, ví dụ `cashew-leaf-api`.
2. Đăng nhập CLI một lần: `hf auth login` (token quyền *write*).
3. Build thư mục deploy (chạy ở máy có đủ file model; script kiểm SHA-256 của bundle):

   ```bash
   python deploy/prepare.py --api-url https://<user>-cashew-leaf-api.hf.space
   ```

4. Đẩy `deploy/_build/space` lên Space (Git LFS cho `.keras`/`.pt`/`.npz`):

   ```bash
   cd deploy/_build/space
   git init -b main && git lfs install
   git add . && git commit -m "Deploy EXP-035 backend"
   git remote add space https://huggingface.co/spaces/<user>/cashew-leaf-api
   git push space main --force
   ```

5. Chờ Space build xong (vài phút) rồi kiểm tra:
   `https://<user>-cashew-leaf-api.hf.space/api/health` → `{"status":"ok",...}` và `/api/model-info` → EXP-035.

## 2. Deploy giao diện lên Vercel

```bash
npm i -g vercel
cd deploy/_build/vercel
vercel login
vercel --prod
```

Khi hỏi: *Set up and deploy?* Yes → *Link to existing project?* No → các câu còn lại để mặc định
(không có build command, output là thư mục hiện tại). Mở URL `*.vercel.app` được in ra.

## Lưu ý

- Space miễn phí tự ngủ khi không dùng; lần gọi đầu có thể chậm 1–2 phút (nạp 5 DenseNet + YOLO).
- Sửa giao diện → chạy lại `prepare.py` rồi `vercel --prod`. Đổi tên Space → chạy lại với `--api-url` mới.
- `deploy/_build/` bị git bỏ qua; không commit model lên GitHub.
- Muốn giới hạn chỉ site Vercel được gọi API: đổi `allow_origins=["*"]` trong `Application/server.py` thành URL Vercel.
