/**
 * Cashew Leaf AI Diagnostic - Frontend Controller
 * Nền trắng chữ đen, giao diện gọn gàng, xếp hạng độ tin cậy từ cao xuống thấp
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropzoneBox = document.getElementById('dropzone-box');
  const fileInput = document.getElementById('file-input');
  const fileStatusTray = document.getElementById('file-status-tray');
  const trayThumbImg = document.getElementById('tray-thumb-img');
  const trayFileName = document.getElementById('tray-file-name');
  const trayFileSize = document.getElementById('tray-file-size');
  const btnTrayRemove = document.getElementById('btn-tray-remove');
  const btnPredict = document.getElementById('btn-predict');
  const btnClearSelection = document.getElementById('btn-clear-selection');

  // Result Area
  const emptyStateView = document.getElementById('empty-state-view');
  const activeResultView = document.getElementById('active-result-view');
  const analyzingOverlay = document.getElementById('analyzing-overlay');
  const statusText = document.getElementById('status-text');

  // Preview elements
  const diagnosisPreviewImg = document.getElementById('diagnosis-preview-img');
  const imgFileName = document.getElementById('img-file-name');
  const imgFileSize = document.getElementById('img-file-size');
  const imageDimBadge = document.getElementById('image-dim-badge');
  const btnZoomImage = document.getElementById('btn-zoom-image');
  const btnReuploadTrigger = document.getElementById('btn-reupload-trigger');

  // Conclusion elements (Bệnh cao nhất trên đầu)
  const conclusionDiseaseName = document.getElementById('conclusion-disease-name');
  const conclusionLatin = document.getElementById('conclusion-latin');
  const conclusionConfidenceValue = document.getElementById('conclusion-confidence-value');
  const conclusionProgressFill = document.getElementById('conclusion-progress-fill');
  const conclusionAdvisoryText = document.getElementById('conclusion-advisory-text');
  const diseaseIconBox = document.getElementById('disease-icon-box');
  const conclusionIcon = document.getElementById('conclusion-icon');

  // Confidence List
  const confidenceList = document.getElementById('confidence-list');

  // Guide elements
  const guideSymptomsText = document.getElementById('guide-symptoms-text');
  const guideManagementList = document.getElementById('guide-management-list');

  // Modal
  const imageModal = document.getElementById('image-modal');
  const modalFullImg = document.getElementById('modal-full-img');
  const btnCloseModal = document.getElementById('btn-close-modal');

  // State
  let currentFile = null;
  let currentPreviewUrl = null;

  const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];

  // ================= 1. FILE UPLOAD & VALIDATION =================

  dropzoneBox.addEventListener('click', () => {
    fileInput.click();
  });

  btnReuploadTrigger.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    handleSelectedFiles(e.target.files);
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzoneBox.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneBox.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzoneBox.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneBox.classList.remove('dragover');
    });
  });

  dropzoneBox.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files) {
      handleSelectedFiles(dt.files);
    }
  });

  function handleSelectedFiles(files) {
    if (!files || files.length === 0) return;

    const file = files[0];

    const isValidType = ALLOWED_TYPES.includes(file.type.toLowerCase()) || 
      /\.(png|jpe?g|webp)$/i.test(file.name);

    if (!isValidType) {
      showToast('Vui lòng chọn tệp hình ảnh (.PNG, .JPG, .JPEG, .WEBP).', 'error');
      resetFileSelection();
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      showToast('Dung lượng ảnh vượt quá 25MB.', 'error');
      resetFileSelection();
      return;
    }

    setFile(file);
    showToast(`Đã chọn ảnh: ${file.name}`, 'success');
  }

  function setFile(file) {
    currentFile = file;

    if (currentPreviewUrl && currentPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(file);

    trayThumbImg.src = currentPreviewUrl;
    trayFileName.textContent = file.name;
    trayFileSize.textContent = formatBytes(file.size);
    fileStatusTray.classList.remove('hidden');

    btnPredict.disabled = false;
  }

  function resetFileSelection() {
    currentFile = null;
    fileInput.value = '';
    if (currentPreviewUrl && currentPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = null;
    fileStatusTray.classList.add('hidden');
    btnPredict.disabled = true;
  }

  btnTrayRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFileSelection();
  });

  btnClearSelection.addEventListener('click', () => {
    resetFileSelection();
    emptyStateView.classList.remove('hidden');
    activeResultView.classList.add('hidden');
    setStatus('Chờ tải ảnh');
    showToast('Đã làm mới.', 'info');
  });

  // ================= 2. PREDICTION EXECUTION =================

  btnPredict.addEventListener('click', async () => {
    if (!currentFile) {
      showToast('Vui lòng chọn 1 hình ảnh trước!', 'warning');
      return;
    }
    await executePrediction(currentFile);
  });

  async function executePrediction(file) {
    const formData = new FormData();
    formData.append('file', file);

    setAnalyzingState(true);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Không thể thực hiện chuẩn đoán.');
      }

      const result = await response.json();
      renderDiagnosisResult(result, currentPreviewUrl, file.name, formatBytes(file.size));
      showToast('Chuẩn đoán hoàn tất!', 'success');
    } catch (err) {
      console.error(err);
      showToast(`Lỗi: ${err.message}`, 'error');
      setStatus('Lỗi');
    } finally {
      setAnalyzingState(false);
    }
  }

  function setAnalyzingState(isAnalyzing) {
    if (isAnalyzing) {
      analyzingOverlay.classList.remove('hidden');
      btnPredict.disabled = true;
      setStatus('Đang phân tích...');
    } else {
      analyzingOverlay.classList.add('hidden');
      btnPredict.disabled = currentFile === null;
    }
  }

  function setStatus(text) {
    if (statusText) statusText.textContent = text;
  }

  // ================= 3. RENDER RESULTS =================

  function renderDiagnosisResult(result, imageUrl, fileName, fileSizeStr) {
    emptyStateView.classList.add('hidden');
    activeResultView.classList.remove('hidden');
    setStatus('Đã chuẩn đoán');

    // 1. Image Preview
    diagnosisPreviewImg.src = imageUrl;
    imgFileName.textContent = fileName;
    imgFileSize.textContent = fileSizeStr;
    imageDimBadge.textContent = `${result.image_meta.width} × ${result.image_meta.height} px`;
    modalFullImg.src = imageUrl;

    // 2. Primary Conclusion (Nhãn cao nhất đặt trên đầu)
    const conclusion = result.conclusion;
    conclusionDiseaseName.textContent = conclusion.label_vn;
    conclusionLatin.textContent = conclusion.scientific_name || '';
    conclusionConfidenceValue.textContent = `${conclusion.confidence_percent.toFixed(2)}%`;
    conclusionProgressFill.style.width = `${conclusion.confidence_percent}%`;
    conclusionAdvisoryText.textContent = conclusion.symptoms || '';

    diseaseIconBox.style.color = conclusion.color_hex;
    diseaseIconBox.style.borderColor = conclusion.color_hex;
    conclusionIcon.className = `fa-solid ${conclusion.icon}`;

    // 3. Render Confidence Breakdown (Xếp hạng từ cao xuống thấp)
    const listDesc = result.probabilities_desc || 
      [...result.probabilities_asc].sort((a, b) => b.confidence - a.confidence);
    renderDescendingConfidenceList(listDesc, conclusion.class_id);

    // 4. Disease Guide Details
    guideSymptomsText.textContent = conclusion.symptoms;
    guideManagementList.innerHTML = '';
    if (conclusion.management && conclusion.management.length > 0) {
      conclusion.management.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        guideManagementList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'Duy trì chăm sóc và theo dõi tán cây định kỳ.';
      guideManagementList.appendChild(li);
    }

    if (window.innerWidth <= 1024) {
      document.getElementById('diagnosis-result-column').scrollIntoView({ behavior: 'smooth' });
    }
  }

  function renderDescendingConfidenceList(classesDesc, winnerClassId) {
    confidenceList.innerHTML = '';

    classesDesc.forEach((item, index) => {
      const isWinner = item.class_id === winnerClassId;
      const rankNumber = index + 1; // 1 (cao nhất trên đầu) -> 5 (thấp nhất)

      const row = document.createElement('div');
      row.className = `confidence-row ${isWinner ? 'is-winner' : ''}`;

      row.innerHTML = `
        <div class="row-top-info">
          <div class="row-label-group">
            <span class="rank-index">#${rankNumber}</span>
            <span class="disease-vn-label">${item.label_vn}</span>
            <span class="disease-en-sub">(${item.class_key})</span>
            ${isWinner ? '<span class="winner-pill-tag"><i class="fa-solid fa-check"></i> KẾT LUẬN</span>' : ''}
          </div>
          <span class="row-conf-value">${item.confidence_percent.toFixed(2)}%</span>
        </div>
        <div class="bar-container">
          <div class="bar-fill" style="width: 0%; background: ${item.color_hex};"></div>
        </div>
      `;

      confidenceList.appendChild(row);

      setTimeout(() => {
        const barFill = row.querySelector('.bar-fill');
        if (barFill) {
          const pct = Math.max(item.confidence_percent, item.confidence_percent > 0 ? 0.6 : 0);
          barFill.style.width = `${pct}%`;
        }
      }, 30 * (index + 1));
    });
  }

  // ================= 4. IMAGE ZOOM MODAL =================

  btnZoomImage.addEventListener('click', () => {
    if (modalFullImg.src) {
      imageModal.classList.remove('hidden');
    }
  });

  btnCloseModal.addEventListener('click', () => {
    imageModal.classList.add('hidden');
  });

  imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) {
      imageModal.classList.add('hidden');
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !imageModal.classList.contains('hidden')) {
      imageModal.classList.add('hidden');
    }
  });

  // ================= 5. TOAST SYSTEM =================

  function showToast(message, type = 'info') {
    const toastStack = document.getElementById('toast-stack');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let iconClass = 'fa-circle-info';
    if (type === 'success') iconClass = 'fa-circle-check';
    if (type === 'warning') iconClass = 'fa-triangle-exclamation';
    if (type === 'error') iconClass = 'fa-circle-exclamation';

    toast.innerHTML = `
      <i class="fa-solid ${iconClass}"></i>
      <span>${message}</span>
    `;

    toastStack.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.2s ease';
      setTimeout(() => toast.remove(), 200);
    }, 3000);
  }

  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }
});
