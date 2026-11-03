/**
 * Cashew Leaf AI Encyclopedia & Diagnostic - Frontend Controller
 * Luồng chuẩn:
 * 1. Trang Chủ (#view-leaf-selector): Chỉ hiển thị danh sách các loại lá cây trồng để chọn.
 *    (Thanh chọn chức năng Phần 1 & Phần 2 hoàn toàn ẩn khi ở Trang Chủ).
 * 2. Sau khi chọn 1 loại lá:
 *    - Hiện thanh chức năng (#leaf-workspace-bar) gồm [Phần 1: Bách Khoa ...] và [Phần 2: Khoanh Vùng & Nhận Diện]
 *      cùng bộ chuyển đổi nhanh giữa các loại lá.
 *    - Nếu là Lá Cây Điều (cashew): Mở Phần 1 (Bách Khoa Lá Điều) hoặc Phần 2 (Khoanh Vùng & Nhận Diện).
 *    - Nếu là các lá khác (coffee, pepper, durian, rubber, rice): Nhảy sang màn hình thông báo "Đang phát triển".
 */

document.addEventListener('DOMContentLoaded', () => {
  // ================= VIEW NAVIGATION ELEMENTS =================
  const views = {
    'view-leaf-selector': document.getElementById('view-leaf-selector'),
    'view-wiki': document.getElementById('view-wiki'),
    'view-under-dev': document.getElementById('view-under-dev'),
    'view-diagnose': document.getElementById('view-diagnose'),
  };

  const leafWorkspaceBar = document.getElementById('leaf-workspace-bar');
  const navBtnSelector = document.getElementById('nav-btn-selector');
  const brandHomeTrigger = document.getElementById('brand-home-trigger');
  const btnWorkspaceBackHome = document.getElementById('btn-workspace-back-home');

  // Function tabs inside Selected Leaf Workspace Bar
  const navBtnWiki = document.getElementById('nav-btn-wiki');
  const navBtnDiagnose = document.getElementById('nav-btn-diagnose');
  const navWikiLabel = document.getElementById('nav-wiki-label');
  const navDiagnoseLabel = document.getElementById('nav-diagnose-label');

  // Buttons inside Wiki (Phần 1)
  const btnSidebarToDiagnose = document.getElementById('btn-sidebar-to-diagnose');
  const btnSidebarBackHome = document.getElementById('btn-sidebar-back-home');
  const btnTopToDiagnose = document.getElementById('btn-top-to-diagnose');
  const btnBottomToDiagnose = document.getElementById('btn-bottom-to-diagnose');

  // Under Development View Elements
  const underDevLeafLabel = document.getElementById('under-dev-leaf-label');
  const underDevFeatureDesc = document.getElementById('under-dev-feature-desc');
  const btnDevGoCashewWiki = document.getElementById('btn-dev-go-cashew-wiki');
  const btnDevBackSelector = document.getElementById('btn-dev-back-selector');

  // Leaf choice cards & dropdown
  const leafChoiceCards = document.querySelectorAll('.leaf-choice-card');
  const quickLeafSelect = document.getElementById('quick-leaf-select');

  // ================= DIAGNOSIS ELEMENTS =================
  const dropzoneBox = document.getElementById('dropzone-box');
  const fileInput = document.getElementById('file-input');
  const fileStatusTray = document.getElementById('file-status-tray');
  const trayThumbImg = document.getElementById('tray-thumb-img');
  const trayFileName = document.getElementById('tray-file-name');
  const trayFileSize = document.getElementById('tray-file-size');
  const btnTrayRemove = document.getElementById('btn-tray-remove');
  const btnPredict = document.getElementById('btn-predict');
  const btnClearSelection = document.getElementById('btn-clear-selection');
  const sampleButtons = document.querySelectorAll('.sample-item-btn');

  // Result Area
  const emptyStateView = document.getElementById('empty-state-view');
  const activeResultView = document.getElementById('active-result-view');
  const analyzingOverlay = document.getElementById('analyzing-overlay');
  const statusText = document.getElementById('status-text');

  // Preview elements
  const diagnosisPreviewImg = document.getElementById('diagnosis-preview-img');
  const bboxDevOverlay = document.getElementById('bbox-dev-overlay');
  const btnToggleBbox = document.getElementById('btn-toggle-bbox');
  const imgFileName = document.getElementById('img-file-name');
  const imgFileSize = document.getElementById('img-file-size');
  const imageDimBadge = document.getElementById('image-dim-badge');
  const btnZoomImage = document.getElementById('btn-zoom-image');
  const btnReuploadTrigger = document.getElementById('btn-reupload-trigger');

  // Conclusion elements
  const conclusionDiseaseName = document.getElementById('conclusion-disease-name');
  const conclusionLatin = document.getElementById('conclusion-latin');
  const conclusionConfidenceValue = document.getElementById('conclusion-confidence-value');
  const conclusionProgressFill = document.getElementById('conclusion-progress-fill');
  const conclusionAdvisoryText = document.getElementById('conclusion-advisory-text');
  const diseaseIconBox = document.getElementById('disease-icon-box');
  const conclusionIcon = document.getElementById('conclusion-icon');

  // Confidence List & Guide
  const confidenceList = document.getElementById('confidence-list');
  const guideSymptomsText = document.getElementById('guide-symptoms-text');
  const guideManagementList = document.getElementById('guide-management-list');

  // Modal
  const imageModal = document.getElementById('image-modal');
  const modalImgTitle = document.getElementById('modal-img-title');
  const modalFullImg = document.getElementById('modal-full-img');
  const btnCloseModal = document.getElementById('btn-close-modal');

  // State
  let currentFile = null;
  let currentPreviewUrl = null;
  let activeLeafId = null;
  let activeLeafPart = 'part1'; // 'part1' (Bách khoa) hoặc 'part2' (Khoanh vùng & Nhận diện)

  const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];

  const LEAF_META = {
    cashew: {
      shortName: 'Lá Điều',
      fullName: 'Lá Cây Điều (Cashew Leaf)',
    },
    coffee: {
      shortName: 'Lá Cà Phê',
      fullName: 'Lá Cây Cà Phê (Coffee Leaf)',
    },
    pepper: {
      shortName: 'Lá Hồ Tiêu',
      fullName: 'Lá Cây Hồ Tiêu (Black Pepper Leaf)',
    },
    durian: {
      shortName: 'Lá Sầu Riêng',
      fullName: 'Lá Cây Sầu Riêng (Durian Leaf)',
    },
    rubber: {
      shortName: 'Lá Cao Su',
      fullName: 'Lá Cây Cao Su (Rubber Leaf)',
    },
    rice: {
      shortName: 'Lá Lúa',
      fullName: 'Lá Cây Lúa (Rice Leaf)',
    },
  };

  // ================= 0. PAGE VIEW NAVIGATION & LEAF SELECTION =================

  function showOnlySection(targetViewId) {
    Object.entries(views).forEach(([id, sectionEl]) => {
      if (!sectionEl) return;
      if (id === targetViewId) {
        sectionEl.classList.remove('hidden');
      } else {
        sectionEl.classList.add('hidden');
      }
    });
  }

  function goToHomeSelector() {
    activeLeafId = null;
    if (leafWorkspaceBar) {
      leafWorkspaceBar.classList.add('hidden');
    }
    if (navBtnSelector) {
      navBtnSelector.classList.add('active');
    }
    showOnlySection('view-leaf-selector');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function openLeafWorkspace(leafId, part = 'part1') {
    activeLeafId = leafId;
    activeLeafPart = part;

    const meta = LEAF_META[leafId] || {
      shortName: 'Lá Cây',
      fullName: 'Loại lá khác',
    };

    // Show the post-leaf-selection workspace bar
    if (leafWorkspaceBar) {
      leafWorkspaceBar.classList.remove('hidden');
    }
    if (navBtnSelector) {
      navBtnSelector.classList.remove('active');
    }
    if (quickLeafSelect) {
      quickLeafSelect.value = leafId;
    }

    // Update the Part 1 & Part 2 button labels for the selected leaf
    if (navWikiLabel) {
      navWikiLabel.textContent = `Phần 1: Bách Khoa ${meta.shortName}`;
    }
    if (navDiagnoseLabel) {
      navDiagnoseLabel.textContent = `Phần 2: Khoanh Vùng & Nhận Diện`;
    }

    // Highlight active part tab inside the selected leaf workspace
    if (navBtnWiki && navBtnDiagnose) {
      navBtnWiki.classList.toggle('active', part === 'part1');
      navBtnDiagnose.classList.toggle('active', part === 'part2');
    }

    if (leafId === 'cashew') {
      if (part === 'part1') {
        showOnlySection('view-wiki');
      } else {
        showOnlySection('view-diagnose');
      }
    } else {
      const partName = part === 'part1'
        ? `Phần 1: Bách Khoa ${meta.shortName}`
        : `Phần 2: Khoanh Vùng & Nhận Diện (${meta.shortName})`;

      if (underDevLeafLabel) {
        underDevLeafLabel.textContent = `Đã chọn: ${meta.fullName} — ${partName}`;
      }
      if (underDevFeatureDesc) {
        underDevFeatureDesc.innerHTML = `Chức năng <strong>${partName}</strong> hiện đang trong quá trình phát triển. Hiện tại hệ thống chỉ hỗ trợ đầy đủ cả 2 phần cho <strong>Lá Cây Điều (Cashew Leaf)</strong>.`;
      }
      showOnlySection('view-under-dev');
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Home triggers
  if (navBtnSelector) {
    navBtnSelector.addEventListener('click', goToHomeSelector);
  }
  if (brandHomeTrigger) {
    brandHomeTrigger.addEventListener('click', goToHomeSelector);
  }
  if (btnWorkspaceBackHome) {
    btnWorkspaceBackHome.addEventListener('click', goToHomeSelector);
  }
  if (btnSidebarBackHome) {
    btnSidebarBackHome.addEventListener('click', goToHomeSelector);
  }
  if (btnDevBackSelector) {
    btnDevBackSelector.addEventListener('click', goToHomeSelector);
  }

  // Leaf cards on Home Page
  leafChoiceCards.forEach(card => {
    card.addEventListener('click', () => {
      const leafId = card.getAttribute('data-leaf-id');
      openLeafWorkspace(leafId, 'part1');
    });
  });

  // Dropdown switcher inside Selected Leaf Workspace Bar
  if (quickLeafSelect) {
    quickLeafSelect.addEventListener('change', (e) => {
      const chosenId = e.target.value;
      openLeafWorkspace(chosenId, activeLeafPart);
    });
  }

  // Part 1 & Part 2 buttons inside Selected Leaf Workspace Bar
  if (navBtnWiki) {
    navBtnWiki.addEventListener('click', () => {
      openLeafWorkspace(activeLeafId || 'cashew', 'part1');
    });
  }
  if (navBtnDiagnose) {
    navBtnDiagnose.addEventListener('click', () => {
      openLeafWorkspace(activeLeafId || 'cashew', 'part2');
    });
  }

  // Internal buttons inside Cashew Wiki to jump to Part 2
  if (btnSidebarToDiagnose) {
    btnSidebarToDiagnose.addEventListener('click', () => openLeafWorkspace('cashew', 'part2'));
  }
  if (btnTopToDiagnose) {
    btnTopToDiagnose.addEventListener('click', () => openLeafWorkspace('cashew', 'part2'));
  }
  if (btnBottomToDiagnose) {
    btnBottomToDiagnose.addEventListener('click', () => openLeafWorkspace('cashew', 'part2'));
  }

  // Button on Under-Development page to switch to Cashew Leaf
  if (btnDevGoCashewWiki) {
    btnDevGoCashewWiki.addEventListener('click', () => openLeafWorkspace('cashew', activeLeafPart));
  }

  // Smooth scroll for Wikipedia TOC links without changing view
  document.querySelectorAll('.toc-link').forEach(link => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        const targetEl = document.querySelector(href);
        if (targetEl) {
          targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });

  // Zoomable images on Wikipedia page
  document.querySelectorAll('.wiki-zoomable-img').forEach(img => {
    img.addEventListener('click', () => {
      modalFullImg.src = img.src;
      if (modalImgTitle) {
        modalImgTitle.textContent = img.alt || 'Chi Tiết Hình Ảnh Lá Điều';
      }
      imageModal.classList.remove('hidden');
    });
  });

  // Toggle Bounding Box preview overlay on Diagnose page
  if (btnToggleBbox && bboxDevOverlay) {
    btnToggleBbox.addEventListener('click', () => {
      const isHidden = bboxDevOverlay.classList.toggle('hidden');
      btnToggleBbox.classList.toggle('active', !isHidden);
      if (!isHidden) {
        showToast('Chế độ khoanh vùng ảnh: Mô hình AI đang trong quá trình huấn luyện.', 'info');
      }
    });
  }

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

  // ================= 2. PREDICTION EXECUTION (UPLOAD & SAMPLES) =================

  btnPredict.addEventListener('click', async () => {
    if (!currentFile) {
      showToast('Vui lòng chọn 1 hình ảnh trước!', 'warning');
      return;
    }
    await executePrediction(currentFile);
  });

  sampleButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const sampleName = btn.getAttribute('data-sample');
      const sampleLabel = btn.getAttribute('data-label') || sampleName;
      if (sampleName) {
        await executeSamplePrediction(sampleName, sampleLabel);
      }
    });
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
      showToast('Nhận diện hoàn tất!', 'success');
    } catch (err) {
      console.error(err);
      showToast(`Lỗi: ${err.message}`, 'error');
      setStatus('Lỗi');
    } finally {
      setAnalyzingState(false);
    }
  }

  async function executeSamplePrediction(sampleName, sampleLabel) {
    setAnalyzingState(true);

    try {
      const response = await fetch(`/api/predict-sample/${encodeURIComponent(sampleName)}`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Không thể chuẩn đoán ảnh mẫu.');
      }

      const result = await response.json();
      const sampleUrl = `/static/samples/${sampleName}?v=cashew`;
      renderDiagnosisResult(result, sampleUrl, sampleName, 'Mẫu Lá Điều Project');
      showToast(`Đã nhận diện: ${sampleLabel}`, 'success');
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
    if (modalImgTitle) {
      modalImgTitle.textContent = `Hình Ảnh Chuẩn Đoán: ${fileName}`;
    }

    // 2. Primary Conclusion
    const conclusion = result.conclusion;
    conclusionDiseaseName.textContent = conclusion.label_vn;
    conclusionLatin.textContent = conclusion.scientific_name || '';
    conclusionConfidenceValue.textContent = `${conclusion.confidence_percent.toFixed(2)}%`;
    conclusionProgressFill.style.width = `${conclusion.confidence_percent}%`;
    conclusionAdvisoryText.textContent = conclusion.symptoms || '';

    diseaseIconBox.style.color = conclusion.color_hex;
    diseaseIconBox.style.borderColor = conclusion.color_hex;
    conclusionIcon.className = `fa-solid ${conclusion.icon}`;

    // 3. Render Confidence Breakdown
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
      const rankNumber = index + 1;

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
