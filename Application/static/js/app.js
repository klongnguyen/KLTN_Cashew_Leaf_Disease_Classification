
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
  const uploadProgress = document.getElementById('upload-progress');
  const sampleButtons = document.querySelectorAll('.sample-item-btn');

  // Result Area
  const emptyStateView = document.getElementById('empty-state-view');
  const activeResultView = document.getElementById('active-result-view');
  const analyzingOverlay = document.getElementById('analyzing-overlay');
  const statusText = document.getElementById('status-text');

  // Preview & Lesion SVG elements
  const diagnosisPreviewImg = document.getElementById('diagnosis-preview-img');
  const lesionSvgOverlay = document.getElementById('lesion-svg-overlay');
  const lesionFilterToolbar = document.getElementById('lesion-filter-toolbar');
  const btnFilterAll = document.getElementById('btn-filter-all');
  const chipCountAll = document.getElementById('chip-count-all');
  const dynamicFilterChips = document.getElementById('dynamic-filter-chips');
  const btnToggleBbox = document.getElementById('btn-toggle-bbox');
  const imgFileName = document.getElementById('img-file-name');
  const imgFileSize = document.getElementById('img-file-size');
  const imageDimBadge = document.getElementById('image-dim-badge');
  const btnZoomImage = document.getElementById('btn-zoom-image');
  const btnReuploadTrigger = document.getElementById('btn-reupload-trigger');

  // Conclusion & Severity elements
  const conclusionCard = document.getElementById('conclusion-card');
  const conclusionModeTag = document.getElementById('conclusion-mode-tag');
  const conclusionDiseaseName = document.getElementById('conclusion-disease-name');
  const conclusionAdvisoryText = document.getElementById('conclusion-advisory-text');
  const nextActionText = document.getElementById('next-action-text');
  const diseaseIconBox = document.getElementById('disease-icon-box');
  const conclusionIcon = document.getElementById('conclusion-icon');
  const detectedDiseasesRow = document.getElementById('detected-diseases-row');

  // Lesion Breakdown elements
  const lesionTotalBadge = document.getElementById('lesion-total-badge');
  const lesionGridTable = document.getElementById('lesion-grid-table');

  // Care guide elements
  const guideProtocolHeading = document.getElementById('guide-protocol-heading');
  const priorityDirectiveBox = document.getElementById('priority-directive-box');
  const guidePriorityNote = document.getElementById('guide-priority-note');
  const guideUrgentList = document.getElementById('guide-urgent-list');
  const guideChemicalBlock = document.getElementById('guide-chemical-block');
  const chemicalRecsWrap = document.getElementById('chemical-recs-wrap');
  const guidePreventionBlock = document.getElementById('guide-prevention-block');
  const guidePreventionList = document.getElementById('guide-prevention-list');
  const btnJumpToWiki = document.getElementById('btn-jump-to-wiki');

  const bboxToggleLabel = document.getElementById('bbox-toggle-label');
  const btnCameraResult = document.getElementById('btn-camera-result');

  // Full-screen viewer
  const imageModal = document.getElementById('image-modal');
  const modalImgTitle = document.getElementById('modal-img-title');
  const modalImgSub = document.getElementById('modal-img-sub');
  const modalFullImg = document.getElementById('modal-full-img');
  const modalBboxLabel = document.getElementById('modal-bbox-label');
  const btnModalToggleBbox = document.getElementById('btn-modal-toggle-bbox');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const viewerSvg = document.getElementById('viewer-svg');
  const viewerStage = document.getElementById('viewer-stage');
  const viewerCanvas = document.getElementById('viewer-canvas');

  // Camera
  const btnOpenCamera = document.getElementById('btn-open-camera');
  const cameraInput = document.getElementById('camera-input');
  const cameraModal = document.getElementById('camera-modal');
  const cameraVideo = document.getElementById('camera-video');
  const cameraSnapshot = document.getElementById('camera-snapshot');
  const cameraCanvas = document.getElementById('camera-canvas');
  const cameraStatus = document.getElementById('camera-status');
  const cameraStatusIcon = document.getElementById('camera-status-icon');
  const cameraStatusText = document.getElementById('camera-status-text');
  const cameraStatusActions = document.getElementById('camera-status-actions');
  const cameraHint = document.getElementById('camera-hint');
  const btnCameraClose = document.getElementById('btn-camera-close');
  const btnCameraShutter = document.getElementById('btn-camera-shutter');
  const btnCameraSwitch = document.getElementById('btn-camera-switch');
  const btnCameraRetake = document.getElementById('btn-camera-retake');
  const btnCameraUse = document.getElementById('btn-camera-use');
  const btnCameraRetry = document.getElementById('btn-camera-retry');
  const btnCameraFallback = document.getElementById('btn-camera-fallback');

  // State
  let currentFile = null;
  let currentPreviewUrl = null;
  let activeLeafId = null;
  let activeLeafPart = 'part1'; // 'part1' (Bách khoa) hoặc 'part2' (Khoanh vùng & Nhận diện)
  let lastDiagnosisResult = null;
  let currentLesionFilter = 'all';
  let isBboxVisible = true;
  let viewerSource = null;      // { src, title } of the diagnosed photo, shown by the full-screen viewer
  let viewerOpen = false;
  let viewerHasBoxes = false;   // the viewer shows the diagnosed photo (with boxes) rather than a wiki photo
  let viewerReturnFocus = null;
  const overlayStack = [];      // open full-screen layers (viewer / camera); the last one is on top
  const diagnoseView = views['view-diagnose'];

  // Lesion overlay state: boxes + server pixel frame, redrawn on every resize
  // isolatedId: when set, only that box is drawn (single-box view)
  const overlayState = { boxes: [], meta: null, focusedId: null, isolatedId: null };
  const previewWrapper = document.getElementById('preview-img-wrapper');
  if (previewWrapper && 'ResizeObserver' in window) {
    // Observe the fixed viewing frame (not the image box, which JS resizes)
    let resizeRaf = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(() => syncOverlayToImage(false));
    }).observe(previewWrapper);
  } else {
    window.addEventListener('resize', () => syncOverlayToImage(false));
  }

  // A surface is a place where the photo is drawn together with its boxes:
  // the preview on the page, or the full-screen viewer.
  const pageSurface = {
    img: diagnosisPreviewImg,
    svg: lesionSvgOverlay,
    canvas: document.getElementById('preview-canvas-box'),
    frame: previewWrapper,
    info: document.getElementById('img-overlay-info'),
    zoomable: true,
  };
  const viewerSurface = {
    img: modalFullImg,
    svg: viewerSvg,
    canvas: viewerCanvas,
    frame: viewerStage,
    info: null,
    zoomable: false,
  };

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

  // On phones/tablets the table of contents starts collapsed so the article is not pushed far down.
  const tocDetails = document.getElementById('wiki-toc-details');
  const tocMedia = window.matchMedia('(max-width: 1024px)');
  if (tocDetails) {
    const syncToc = () => { tocDetails.open = !tocMedia.matches; };
    syncToc();
    if (tocMedia.addEventListener) tocMedia.addEventListener('change', syncToc);
    else if (tocMedia.addListener) tocMedia.addListener(syncToc);
  }

  // Smooth scroll for Wikipedia TOC links without changing view
  document.querySelectorAll('.toc-link').forEach(link => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        const targetEl = document.querySelector(href);
        if (tocDetails && tocMedia.matches) tocDetails.open = false;
        if (targetEl) {
          targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });

  // Zoomable images on Wikipedia page (full-screen viewer, no boxes)
  document.querySelectorAll('.wiki-zoomable-img').forEach(img => {
    img.addEventListener('click', () => {
      openViewer({ src: img.src, title: img.alt || 'Chi Tiết Hình Ảnh Lá Điều', withBoxes: false });
    });
  });

  // Show/hide the boxes — one switch shared by the page preview and the full-screen viewer
  function setBboxVisible(visible) {
    isBboxVisible = visible;
    const display = visible ? 'block' : 'none';
    if (lesionSvgOverlay) lesionSvgOverlay.style.display = display;
    if (viewerSvg) viewerSvg.style.display = display;
    [btnToggleBbox, btnModalToggleBbox].forEach(btn => {
      if (!btn) return;
      btn.classList.toggle('active', visible);
      btn.setAttribute('aria-pressed', String(visible));
    });
    const label = visible ? 'Ẩn khung' : 'Hiện khung';
    if (bboxToggleLabel) bboxToggleLabel.textContent = label;
    if (modalBboxLabel) modalBboxLabel.textContent = label;
    // Labels are measured while drawn; redraw so they are sized right once visible again
    if (visible) syncOverlayToImage(false);
  }

  if (btnToggleBbox) btnToggleBbox.addEventListener('click', () => setBboxVisible(!isBboxVisible));
  if (btnModalToggleBbox) btnModalToggleBbox.addEventListener('click', () => setBboxVisible(!isBboxVisible));

  // Filter All Lesions Button
  if (btnFilterAll) {
    btnFilterAll.addEventListener('click', () => {
      applyLesionFilter('all');
    });
  }

  // Jump to Wikipedia CTA Button
  if (btnJumpToWiki) {
    btnJumpToWiki.addEventListener('click', () => {
      activeLeafPart = 'part1';
      if (navBtnWiki) navBtnWiki.click();
      if (lastDiagnosisResult && lastDiagnosisResult.conclusion) {
        const topLabel = lastDiagnosisResult.conclusion.label;
        let targetSecId = 'wiki-sec-diseases';
        if (topLabel === 'anthracnose') targetSecId = 'wiki-disease-anthracnose';
        else if (topLabel === 'leaf_miner') targetSecId = 'wiki-disease-leafminer';
        else if (topLabel === 'red_rust') targetSecId = 'wiki-disease-redrust';
        else if (topLabel === 'healthy') targetSecId = 'wiki-disease-healthy';
        setTimeout(() => {
          const el = document.getElementById(targetSecId);
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }, 150);
      }
    });
  }

  // ================= 1. FILE UPLOAD & VALIDATION =================

  dropzoneBox.addEventListener('click', () => {
    fileInput.click();
  });

  btnReuploadTrigger.addEventListener('click', () => {
    fileInput.value = '';
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
    diagnoseView.classList.remove('has-result');
    diagnoseView.classList.add('has-selection');
    activeResultView.classList.add('hidden');
    emptyStateView.classList.remove('hidden');

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
    diagnoseView.classList.remove('has-selection');
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
    diagnoseView.classList.remove('has-result');
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
        const errorData = await response.json().catch(() => ({}));
        console.error('Predict failed:', response.status, errorData.detail);
        throw new Error(friendlyError(response.status, errorData.detail, 'Không thể thực hiện chuẩn đoán.'));
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
        const errorData = await response.json().catch(() => ({}));
        console.error('Predict sample failed:', response.status, errorData.detail);
        throw new Error(friendlyError(response.status, errorData.detail, 'Không thể chuẩn đoán ảnh mẫu.'));
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
      uploadProgress.classList.remove('hidden');
      btnPredict.disabled = true;
      setStatus('Đang phân tích...');
    } else {
      analyzingOverlay.classList.add('hidden');
      uploadProgress.classList.add('hidden');
      btnPredict.disabled = currentFile === null;
    }
  }

  function setStatus(text) {
    if (statusText) statusText.textContent = text;
  }

  // ================= 3. RENDER RESULTS =================

  function renderDiagnosisResult(result, imageUrl, fileName, fileSizeStr) {
    lastDiagnosisResult = result;
    currentLesionFilter = 'all';

    emptyStateView.classList.add('hidden');
    activeResultView.classList.remove('hidden');
    diagnoseView.classList.add('has-result');
    diagnoseView.classList.remove('has-selection');
    document.querySelectorAll('#active-result-view details').forEach(details => { details.open = false; });
    setStatus('Đã chuẩn đoán');

    // 1. Image Preview & Dimensions
    imgFileName.textContent = fileName;
    imgFileSize.textContent = fileSizeStr;
    imageDimBadge.textContent = `${result.image_meta.width} × ${result.image_meta.height} px`;
    viewerSource = { src: imageUrl, title: fileName };

    // 2. Draw Interactive SVG Bounding Boxes (after the image has decoded,
    //    so naturalWidth/naturalHeight are known and the overlay can be pinned
    //    to the exact rendered pixel rectangle of the image)
    overlayState.boxes = result.boxes || [];
    overlayState.meta = result.image_meta;
    overlayState.focusedId = null;
    overlayState.isolatedId = null;
    const cb = document.getElementById('preview-canvas-box');
    if (cb) cb.style.transform = '';
    diagnosisPreviewImg.classList.remove('is-synced', 'raw-orientation');
    if (lesionSvgOverlay) lesionSvgOverlay.innerHTML = '';
    diagnosisPreviewImg.onload = () => syncOverlayToImage(true);
    diagnosisPreviewImg.src = imageUrl;
    if (diagnosisPreviewImg.complete && diagnosisPreviewImg.naturalWidth > 0) {
      syncOverlayToImage(true);
    }

    // 2b. Only a critical engine warning (fallback model in use) is shown;
    //     the only input-quality note left is the image-size warning.
    renderInputWarnings(result);

    // 3. Render Lesion Filter Toolbar
    renderFilterChips(result.diseases_detected || [], result.total_lesion_boxes || 0, result.boxes || []);
    const hasBoxes = (result.boxes || []).length > 0;
    document.getElementById('image-preview-title').textContent = hasBoxes ? 'Vùng nghi bệnh trên lá' : 'Ảnh lá đã kiểm tra';
    btnToggleBbox.classList.toggle('hidden', !hasBoxes);
    if (result.conclusion.ood_rejected) {
      document.getElementById('lesion-summary-line').textContent = 'Ảnh không phù hợp để khoanh vùng bệnh lá điều.';
    }

    // 4. Primary Conclusion & Multi-Disease Mode
    const conclusion = result.conclusion;
    const conclusionCard = document.getElementById('conclusion-card');
    conclusionDiseaseName.textContent = conclusion.label === 'not_cashew_leaf' ? 'Ảnh không phải lá điều' : conclusion.label_vn;

    const actions = ((result.treatment_plan && result.treatment_plan.urgent_actions) || []).map(userText).filter(Boolean);
    const suggestedAction = conclusion.ood_rejected
      ? 'Hãy chọn ảnh lá điều rõ hơn để nhận diện lại. Ảnh này không phù hợp để kết luận bệnh.'
      : conclusion.abstain
        ? 'Chụp lại ảnh rõ hơn hoặc hỏi cán bộ chuyên môn trước khi xử lý. Chưa phun thuốc diện rộng.'
        : conclusion.label === 'healthy'
          ? 'Tiếp tục theo dõi lá và kiểm tra vườn định kỳ.'
          : actions[0] || 'Kiểm tra lại các vùng được đánh dấu trên lá và theo dõi diễn biến.';
    const uncertain = !conclusion.ood_rejected && !conclusion.abstain && conclusion.label !== 'healthy'
      && conclusion.confidence_percent != null && conclusion.confidence_percent < 70;
    nextActionText.textContent = uncertain
      ? `Kết quả còn chưa chắc chắn; hãy đối chiếu vết trên lá. ${suggestedAction}`
      : suggestedAction;

    // Populate Indicator 1: Whole-Leaf Confidence
    const globalConfEl = document.getElementById('global-conf-score');
    if (globalConfEl) {
      globalConfEl.textContent = (conclusion.confidence_percent === null || conclusion.confidence_percent === undefined)
        ? '—'
        : `${conclusion.confidence_percent}%`;
      globalConfEl.parentElement.classList.toggle('hidden', conclusion.ood_rejected || conclusion.abstain || conclusion.confidence_percent == null);
    }

    const ind1Title = document.getElementById('indicator1-title');
    const confSub = document.getElementById('global-conf-sub');
    if (conclusion.ood_rejected) {
      if (ind1Title) ind1Title.textContent = 'Chưa thể nhận diện';
      if (confSub) confSub.textContent = 'Không áp dụng';
    } else if (conclusion.confidence_source === 'box') {
      if (ind1Title) ind1Title.textContent = conclusion.is_multi_disease ? 'Phát hiện nhiều bệnh' : 'Kết quả nhận diện';
      if (confSub) confSub.textContent = 'Độ tin cậy vùng rõ nhất';
    } else {
      if (ind1Title) ind1Title.textContent = conclusion.abstain ? 'Cần kiểm tra thêm' : 'Kết quả nhận diện';
      if (confSub) confSub.textContent = 'Độ tin cậy toàn ảnh';
    }

    if (conclusion.abstain) {
      if (conclusionCard) conclusionCard.classList.add('is-abstain');
      conclusionModeTag.innerHTML = '<i class="fa-solid fa-user-doctor"></i> Cần kiểm tra thêm';
      conclusionModeTag.className = 'badge-conclusion-tag abstain-mode';

      let top2Html = '';
      if (result.top2 && result.top2.length >= 2) {
        const t1 = result.top2[0];
        const t2 = result.top2[1];
        const pDesc = result.probabilities_desc || [];
        const item1 = pDesc.find(p => p.class_key === t1[0]) || {};
        const item2 = pDesc.find(p => p.class_key === t2[0]) || {};
        const name1 = item1.label_vn || t1[0];
        const name2 = item2.label_vn || t2[0];
        top2Html = `<div style="margin-top: 0.5rem; padding: 0.45rem 0.65rem; background: #fef08a; border-radius: 3px; font-size: 0.76rem; color: #713f12; border: 1px solid #facc15;">
          <i class="fa-solid fa-scale-balanced" style="margin-right: 0.35rem;"></i>
          <strong>Phân vân giữa 2 khả năng:</strong> 1. ${name1} (${(t1[1] * 100).toFixed(1)}%) &bull; 2. ${name2} (${(t2[1] * 100).toFixed(1)}%)
        </div>`;
      }
      conclusionAdvisoryText.textContent = userText(conclusion.symptoms);
      if (top2Html) conclusionAdvisoryText.insertAdjacentHTML('beforeend', top2Html);
    } else {
      if (conclusionCard) conclusionCard.classList.remove('is-abstain');
      conclusionAdvisoryText.textContent = userText(conclusion.symptoms);

      if (conclusion.is_multi_disease) {
        conclusionModeTag.innerHTML = '<i class="fa-solid fa-viruses"></i> Nhiều bệnh';
        conclusionModeTag.className = 'badge-conclusion-tag multi-mode';
      } else if (conclusion.label === 'healthy') {
        conclusionModeTag.innerHTML = '<i class="fa-solid fa-shield-halved"></i> Lá khỏe';
        conclusionModeTag.className = 'badge-conclusion-tag';
      } else if (conclusion.label === 'not_cashew_leaf') {
        conclusionModeTag.innerHTML = conclusion.ood_rejected
          ? '<i class="fa-solid fa-ban"></i> Ảnh không phù hợp'
          : '<i class="fa-solid fa-triangle-exclamation"></i> Ảnh không phù hợp';
        conclusionModeTag.className = 'badge-conclusion-tag';
      } else {
        conclusionModeTag.innerHTML = '<i class="fa-solid fa-check"></i> Đã nhận diện';
        conclusionModeTag.className = 'badge-conclusion-tag';
      }
    }

    diseaseIconBox.style.color = conclusion.color_hex;
    diseaseIconBox.style.borderColor = conclusion.color_hex;
    conclusionIcon.className = `fa-solid ${conclusion.icon}`;
    conclusionModeTag.classList.toggle('hidden', !conclusion.abstain && !conclusion.is_multi_disease && conclusion.label !== 'healthy' && conclusion.label !== 'not_cashew_leaf');
    conclusionCard.classList.toggle('is-healthy', conclusion.label === 'healthy');
    conclusionCard.classList.toggle('is-rejected', !!conclusion.ood_rejected);

    // 5. Render Detected Diseases Row
    renderDetectedDiseasesRow(conclusion.is_multi_disease ? (result.diseases_detected || []) : []);
    document.getElementById('lesion-breakdown-section').classList.toggle('hidden', !(result.boxes || []).length);

    // 6. Render Lesion Breakdown Table & Thumbnails
    renderLesionBreakdown(result.boxes || []);

    // 7. Render the care guide shown under the conclusion
    renderTreatmentProtocol(result.treatment_plan, conclusion);

    if (window.innerWidth <= 1024) {
      document.getElementById('diagnosis-result-column').scrollIntoView({ behavior: 'smooth' });
    }
  }

  /**
   * Pins the <img> and the SVG overlay onto the exact pixel rectangle that the
   * image occupies inside the (square) preview wrapper, then redraws boxes.
   *
   * Coordinate frame: boxes arrive in the server's pixel frame (image_meta).
   * The SVG viewBox is that frame and the SVG element is sized to the image's
   * rendered rectangle, so 1 server pixel maps to exactly 1 image pixel on
   * screen with no letterbox offset and no aspect distortion.
   *
   * If the browser shows the photo EXIF-rotated while the server measured the
   * raw (unrotated) frame, width/height come out swapped. In that case the
   * preview is switched to raw orientation so both frames are identical.
   */
  function syncOverlayToImage(isNewImage) {
    layoutSurface(pageSurface, isNewImage);
    if (viewerOpen) layoutSurface(viewerSurface, isNewImage);
  }

  /**
   * Lays out one surface (page preview or full-screen viewer): fits the photo to
   * its frame, then draws the boxes in the server's pixel frame on top of it.
   * The viewer shows boxes only when it was opened for the diagnosed photo.
   */
  function layoutSurface(s, isNewImage) {
    const { img, svg, canvas, frame } = s;
    if (!img) return;

    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    if (!nw || !nh) return;

    const meta = (s.zoomable || viewerHasBoxes) ? overlayState.meta : null;
    const mw = (meta && meta.width) || nw;
    const mh = (meta && meta.height) || nh;

    if (meta && isNewImage && !img.classList.contains('raw-orientation') &&
        nw !== nh && nw === mh && nh === mw) {
      img.classList.add('raw-orientation');
      requestAnimationFrame(() => layoutSurface(s, false));
      return;
    }

    // Fit the image to the viewing frame (contain). Small photos are UPSCALED
    // so lesions are visible; the canvas box follows the <img> size exactly,
    // so the SVG overlay stays aligned 1:1 with the image pixels.
    if (frame) {
      const availW = frame.clientWidth;
      const availH = frame.clientHeight - (s.info ? s.info.offsetHeight : 0);
      if (availW > 0 && availH > 0) {
        const fit = Math.min(availW / nw, availH / nh);
        img.style.width = `${Math.floor(nw * fit)}px`;
        img.style.height = `${Math.floor(nh * fit)}px`;
      }
    }

    if (!svg) return;
    if (!meta) {
      svg.innerHTML = '';
      return;
    }

    // The canvas wraps <img> directly, so the SVG overlay (width/height: 100%)
    // has the EXACT same rendered size as the visible image. Its viewBox is the
    // original coordinate space from the server:
    svg.setAttribute('viewBox', `0 0 ${mw} ${mh}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    const renderedW = img.clientWidth || (canvas ? canvas.clientWidth : nw);
    const zoom = s.zoomable ? computeZoom(canvas, mw) : { s: 1, tx: 0, ty: 0 };
    // Divide by the zoom so labels keep a constant on-screen size when zoomed
    const unitsPerPx = renderedW > 0 ? (mw / renderedW) / zoom.s : 1;

    drawLesionBoxes(overlayState.boxes, mw, mh, unitsPerPx, svg);
    if (canvas && s.zoomable) {
      canvas.style.transform = zoom.s === 1 ? '' : `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.s})`;
    }
  }

  /**
   * Single-box view: zoom of the canvas (image + overlay together) that puts
   * the isolated box in the middle of the frame; identity when no box is
   * isolated. Only a CSS transform is used, so box coordinates never change.
   */
  function computeZoom(canvasBox, frameW) {
    const none = { s: 1, tx: 0, ty: 0 };
    const box = overlayState.isolatedId !== null
      ? overlayState.boxes.find(b => b.id === overlayState.isolatedId)
      : null;
    if (!box || !canvasBox) return none;
    const W = canvasBox.offsetWidth, H = canvasBox.offsetHeight;
    if (!W || !H) return none;
    const sx = W / (frameW || W);
    const [x1, y1, x2, y2] = box.xyxy;
    const bw = Math.max((x2 - x1) * sx, 1), bh = Math.max((y2 - y1) * sx, 1);
    // box should fill ~35% of the frame (context stays visible); zoom 1x..3x
    const s = Math.max(1, Math.min(3, 0.35 * Math.min(W / bw, H / bh)));
    const cx = ((x1 + x2) / 2) * sx, cy = ((y1 + y2) / 2) * sx;
    // keep the scaled image covering the canvas (no empty margin inside it)
    const tx = Math.min(0, Math.max(W - W * s, W / 2 - cx * s));
    const ty = Math.min(0, Math.max(H - H * s, H / 2 - cy * s));
    return { s, tx, ty };
  }

  // Re-sync redraws the labels for the new zoom level and applies the transform
  function applyZoom() {
    syncOverlayToImage(false);
  }

  /**
   * @param {number} frameW,frameH  server pixel frame (= SVG viewBox)
   * @param {number} unitsPerPx     frame units per screen pixel, used to keep
   *                                labels a constant on-screen size
   */
  function drawLesionBoxes(boxes, frameW, frameH, unitsPerPx, svg = lesionSvgOverlay) {
    if (!svg) return;
    svg.innerHTML = '';
    if (!boxes || boxes.length === 0) return;

    const k = unitsPerPx > 0 ? unitsPerPx : 1;
    const fontSize = 11 * k;
    const tagH = 18 * k;
    const padX = 5 * k;
    const svgNS = 'http://www.w3.org/2000/svg';
    const hasSeveralDiseases = new Set(boxes.map(box => box.cls)).size > 1;

    boxes.forEach(box => {
      const [x1, y1, x2, y2] = box.xyxy;
      const bw = Math.max(x2 - x1, 2 * k);
      const bh = Math.max(y2 - y1, 2 * k);

      const g = document.createElementNS(svgNS, 'g');
      g.setAttribute('class', 'lesion-box-group');
      g.setAttribute('data-box-id', box.id);
      g.setAttribute('data-cls', box.cls);

      // Rectangle (exact server coordinates)
      const rect = document.createElementNS(svgNS, 'rect');
      rect.setAttribute('class', 'lesion-rect');
      rect.setAttribute('x', x1);
      rect.setAttribute('y', y1);
      rect.setAttribute('width', bw);
      rect.setAttribute('height', bh);
      rect.setAttribute('stroke', box.color_hex);
      rect.setAttribute('fill', box.color_hex);

      // Keep labels short on the photo; scores and model details live below.
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('class', 'lesion-tag-text');
      text.setAttribute('font-size', fontSize);
      const disease = (lastDiagnosisResult.diseases_detected || []).find(item => item.class_key === box.cls);
      text.textContent = hasSeveralDiseases
        ? `#${box.id} ${disease?.short_name || disease?.label_vn || box.label_vn}`
        : `#${box.id}`;

      const tagBg = document.createElementNS(svgNS, 'rect');
      tagBg.setAttribute('class', 'lesion-tag-bg');
      tagBg.setAttribute('rx', 2 * k);
      tagBg.setAttribute('ry', 2 * k);
      tagBg.setAttribute('height', tagH);
      tagBg.setAttribute('fill', box.color_hex);

      g.appendChild(rect);
      g.appendChild(tagBg);
      g.appendChild(text);
      svg.appendChild(g);

      let textLen = 0;
      try { textLen = text.getComputedTextLength(); } catch (_) { textLen = 0; }
      if (!textLen) textLen = text.textContent.length * 6.2 * k;
      const tagW = Math.min(textLen + padX * 2, frameW);

      // Keep the tag inside the image: above the box if room, else inside it
      const tagX = Math.max(0, Math.min(x1, frameW - tagW));
      const tagY = y1 >= tagH ? y1 - tagH : Math.min(y1, frameH - tagH);

      tagBg.setAttribute('x', tagX);
      tagBg.setAttribute('y', tagY);
      tagBg.setAttribute('width', tagW);
      text.setAttribute('x', tagX + padX);
      text.setAttribute('y', tagY + tagH / 2);

      const showTag = boxes.length <= 12 || overlayState.focusedId === box.id;
      tagBg.style.display = showTag ? '' : 'none';
      text.style.display = showTag ? '' : 'none';

      if (overlayState.focusedId === box.id) g.classList.add('is-focused');

      // Click on a box -> show ONLY that box (click again to show all)
      g.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleIsolate(box.id, false, svg === viewerSvg);
      });
    });

    applyLesionVisibility();
    const focused = svg.querySelector('.lesion-box-group.is-focused');
    if (focused) svg.appendChild(focused);
  }

  /**
   * Disease count line + one chip per disease class present among the drawn
   * boxes (count = boxes of that class). Built from the boxes themselves so
   * every class that is drawn can be filtered.
   */
  function renderFilterChips(diseasesDetected, totalBoxes, boxes) {
    if (!dynamicFilterChips) return;
    dynamicFilterChips.innerHTML = '';
    boxes = boxes || [];

    if (chipCountAll) chipCountAll.textContent = boxes.length || totalBoxes || 0;
    if (btnFilterAll) {
      btnFilterAll.classList.add('active');
      btnFilterAll.disabled = boxes.length === 0;
    }

    const groups = [];
    boxes.forEach(b => {
      let g = groups.find(x => x.cls === b.cls);
      if (!g) {
        const d = (diseasesDetected || []).find(x => x.class_key === b.cls) || {};
        g = { cls: b.cls, name: d.short_name || d.label_vn || b.label_vn, color: b.color_hex || d.color_hex, count: 0 };
        groups.push(g);
      }
      g.count += 1;
    });
    groups.sort((a, b) => b.count - a.count);

    const summary = document.getElementById('lesion-summary-line');
    if (summary) {
      summary.innerHTML = boxes.length === 0
        ? '<i class="fa-solid fa-circle-info"></i> Không có vùng bệnh nào được khoanh trên ảnh này.'
        : `<i class="fa-solid fa-crosshairs"></i> <strong>${boxes.length} vùng nghi bệnh</strong> được đánh dấu trên ảnh.`;
    }
    lesionFilterToolbar.classList.toggle('has-class-filters', groups.length > 1);

    groups.forEach(g => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'lesion-filter-chip';
      chip.setAttribute('data-filter', g.cls);
      chip.innerHTML = `
        <span class="dot-indicator" style="background: ${g.color}; width: 8px; height: 8px; border-radius: 50%;"></span>
        <span>${escapeHtml(g.name)}</span>
        <span class="chip-count">${g.count}</span>
      `;
      chip.addEventListener('click', () => applyLesionFilter(g.cls));
      dynamicFilterChips.appendChild(chip);
    });
  }

  /** Class filter. Selecting a class (or "all") leaves single-box view. */
  function applyLesionFilter(filterCls) {
    currentLesionFilter = filterCls;
    overlayState.isolatedId = null;
    overlayState.focusedId = null;

    if (btnFilterAll) btnFilterAll.classList.toggle('active', filterCls === 'all');
    if (dynamicFilterChips) {
      dynamicFilterChips.querySelectorAll('.lesion-filter-chip').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-filter') === filterCls);
      });
    }
    applyLesionVisibility();
    applyZoom();
  }

  /** Single source of truth for which boxes / cards are shown. */
  function applyLesionVisibility() {
    const iso = overlayState.isolatedId;
    const visible = (cls, id) => iso !== null ? id === iso : (currentLesionFilter === 'all' || cls === currentLesionFilter);

    [lesionSvgOverlay, viewerSvg].forEach(svg => {
      if (!svg) return;
      svg.querySelectorAll('.lesion-box-group').forEach(g => {
        const id = parseInt(g.getAttribute('data-box-id'), 10);
        g.classList.toggle('is-hidden', !visible(g.getAttribute('data-cls'), id));
        g.classList.toggle('is-focused', iso !== null && id === iso);
      });
    });
    if (lesionGridTable) {
      lesionGridTable.querySelectorAll('.lesion-card').forEach(card => {
        const id = parseInt(card.getAttribute('data-box-id'), 10);
        const cls = card.getAttribute('data-cls');
        // cards follow the class filter; the isolated card is outlined
        card.style.display = (currentLesionFilter === 'all' || cls === currentLesionFilter || id === iso) ? 'flex' : 'none';
        card.classList.toggle('is-isolated', id === iso);
        card.classList.toggle('is-active', id === iso);
      });
    }
    const pill = document.getElementById('btn-clear-isolate');
    if (pill) {
      pill.classList.toggle('hidden', iso === null);
      const label = document.getElementById('isolate-label');
      const b = iso !== null ? overlayState.boxes.find(x => x.id === iso) : null;
      if (label && b) label.textContent = `Đang xem vùng #${b.id} — ${b.label_vn} · Hiện tất cả`;
    }
  }

  /**
   * Single-box view. fromCard: the click came from the breakdown table, so the
   * image is scrolled into view; a click on the image scrolls to the card.
   */
  function toggleIsolate(boxId, fromCard, fromViewer) {
    overlayState.isolatedId = overlayState.isolatedId === boxId ? null : boxId;
    overlayState.focusedId = overlayState.isolatedId;
    applyLesionVisibility();
    applyZoom();
    if (overlayState.isolatedId === null || fromViewer) return;
    if (fromCard && previewWrapper) {
      previewWrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (lesionGridTable) {
      document.getElementById('lesion-breakdown-section').open = true;
      const card = lesionGridTable.querySelector(`.lesion-card[data-box-id="${boxId}"]`);
      if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function clearIsolate() {
    if (overlayState.isolatedId === null) return;
    overlayState.isolatedId = null;
    overlayState.focusedId = null;
    applyLesionVisibility();
    applyZoom();
  }

  const btnClearIsolate = document.getElementById('btn-clear-isolate');
  if (btnClearIsolate) btnClearIsolate.addEventListener('click', clearIsolate);
  // Click on empty image area also leaves single-box view
  if (previewWrapper) previewWrapper.addEventListener('click', clearIsolate);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlayStack.length === 0) clearIsolate();
  });

  /**
   * Text that comes from the server is shown to farmers, so it must never name the
   * internal models or their scores. Any sentence that does is dropped (the server
   * wording is already plain; this is the safety net for every engine / version).
   */
  function userText(text) {
    if (text === null || text === undefined) return '';
    const jargon = /yolo|densenet|knn|stacker|energy|ensemble|backbone|checkpoint|exp-\d|legacy|traceback|exception|\b\w+error\b/i;
    const sentences = String(text).match(/[\s\S]+?(?:[.!?]+(?=\s|$)|$)\s*/g) || [];
    return sentences.filter(part => !jargon.test(part)).join('').trim();
  }

  /** Error shown in a toast: server faults get a plain message, validation messages pass through. */
  function friendlyError(status, detail, fallback) {
    if (status >= 500) return 'Hệ thống đang gặp sự cố khi nhận diện. Vui lòng thử lại sau ít phút.';
    return userText(detail) || fallback;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function renderInputWarnings(result) {
    const card = document.getElementById('conclusion-card');
    if (!card) return;
    let box = document.getElementById('input-warning-box');
    if (!box) {
      box = document.createElement('div');
      box.id = 'input-warning-box';
      box.className = 'input-warning-box';
    }
    // Keep warnings beside the action, without displacing the main answer.
    const signs = document.getElementById('signs-details');
    if (box.parentNode !== card) card.insertBefore(box, signs);
    const items = [];
    const engineWarning = userText(result.model_meta && result.model_meta.engine_warning);
    if (engineWarning) items.push({ icon: 'fa-server', text: engineWarning, strong: true });
    const sizeWarnings = (result.quality && result.quality.warnings) || [];
    if (sizeWarnings.length) items.push({ icon: 'fa-camera', text: 'Ảnh khá nhỏ, có thể bỏ sót vết bệnh. Hãy chụp cận cảnh rõ hơn nếu cần.', strong: false });
    const gateWarning = result.ood_gate && result.ood_gate.warning;
    if (gateWarning && !result.conclusion.ood_rejected) {
      items.push({ icon: 'fa-image', text: 'Ảnh khác với ảnh thường gặp; hãy kiểm tra lại kết quả hoặc chụp ảnh khác.', strong: false });
    }
    if (items.length === 0) {
      box.classList.add('hidden');
      box.innerHTML = '';
      return;
    }
    box.classList.remove('hidden');
    box.innerHTML = `
      <div class="input-warning-title"><i class="fa-solid fa-triangle-exclamation"></i> ${engineWarning ? 'Cảnh báo hệ thống' : 'Lưu ý về kết quả'}</div>
      <ul>${items.map(i => `<li class="${i.strong ? 'is-strong' : ''}"><i class="fa-solid ${i.icon}"></i> ${escapeHtml(i.text)}</li>`).join('')}</ul>
    `;
  }

  function renderDetectedDiseasesRow(diseasesDetected) {
    if (!detectedDiseasesRow) return;
    detectedDiseasesRow.innerHTML = '';
    detectedDiseasesRow.classList.toggle('hidden', !diseasesDetected || diseasesDetected.length < 2);

    if (!diseasesDetected || diseasesDetected.length === 0) return;

    diseasesDetected.forEach(d => {
      const pill = document.createElement('div');
      pill.className = 'disease-pill-chip';
      pill.innerHTML = `
        <span class="dot-indicator" style="background: ${d.color_hex};"></span>
        <span>${d.label_vn}: <strong>${d.box_count} vùng khoanh</strong></span>
      `;
      detectedDiseasesRow.appendChild(pill);
    });
  }

  function renderLesionBreakdown(boxes) {
    if (!lesionGridTable) return;
    lesionGridTable.innerHTML = '';

    if (lesionTotalBadge) {
      lesionTotalBadge.textContent = `${boxes.length} vùng`;
    }

    if (!boxes || boxes.length === 0) {
      lesionGridTable.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 1rem; text-align: center; color: var(--text-muted); font-size: 0.82rem; background: var(--bg-subtle); border-radius: 3px;">
          <i class="fa-solid fa-circle-check" style="color: #10b981; margin-right: 0.35rem;"></i>
          Không phát hiện vùng tổn thương khu trú nào trên chiếc lá này.
        </div>
      `;
      return;
    }

    boxes.forEach(box => {
      const card = document.createElement('div');
      card.className = 'lesion-card';
      card.setAttribute('data-box-id', box.id);
      card.setAttribute('data-cls', box.cls);

      card.innerHTML = `
        <div class="lesion-thumb-wrap">
          <img src="${box.crop_base64}" alt="Vết bệnh #${box.id}">
        </div>
        <div class="lesion-card-info">
          <div class="lesion-card-top">
            <span class="lesion-id-label">Ổ bệnh #${box.id}</span>
            <span class="lesion-disease-pill" style="background: ${box.color_hex};">${box.label_vn}</span>
          </div>
          <div class="lesion-dim-row">
            <span>Kích thước: <strong>${box.width_px} × ${box.height_px} px</strong></span>
            <span>&bull;</span>
            <span>Chiếm: <strong>${box.area_percent}% ảnh</strong></span>
          </div>
          <div class="lesion-card-bottom">
            ${(box.conf_percent === null || box.conf_percent === undefined)
              ? `<span class="lesion-conf-score" title="Vùng tìm theo màu sắc, không có điểm tin cậy của mô hình"><i class="fa-solid fa-circle-question"></i> Vùng nghi ngờ (chưa chấm điểm)</span>`
              : `<span class="lesion-conf-score"><i class="fa-solid fa-bullseye"></i> Độ tin cậy: <strong>${box.conf_percent}%</strong></span>`}
            <button type="button" class="btn-locate-lesion" title="Chỉ hiện vùng này trên ảnh (bấm lại để hiện tất cả)">
              <i class="fa-solid fa-crosshairs"></i> Chỉ xem vùng này
            </button>
          </div>
        </div>
      `;

      card.addEventListener('click', () => {
        toggleIsolate(box.id, true);
      });

      lesionGridTable.appendChild(card);
    });
  }

  /**
   * Care guide shown under the conclusion. Only blocks that have content are
   * displayed, and wording follows the situation: normal care, "ask an expert"
   * (low confidence), or "retake the photo" (not a cashew leaf).
   */
  function renderTreatmentProtocol(treatmentPlan, conclusion) {
    if (!treatmentPlan) return;
    conclusion = conclusion || {};

    const mode = conclusion.ood_rejected ? 'retake' : conclusion.abstain ? 'refer' : 'care';
    const copy = {
      care: { heading: 'Hướng dẫn chăm sóc chi tiết', urgent: 'Chăm sóc và vệ sinh vườn', icon: 'fa-scissors' },
      refer: { heading: 'Khuyến nghị tiếp theo', urgent: 'Việc cần làm', icon: 'fa-list-check' },
      retake: { heading: 'Hướng dẫn chụp lại ảnh', urgent: 'Cách chụp ảnh rõ hơn', icon: 'fa-camera' },
    }[mode];

    if (guideProtocolHeading) guideProtocolHeading.textContent = copy.heading;
    const urgentLabel = document.getElementById('guide-urgent-label');
    if (urgentLabel) {
      urgentLabel.textContent = copy.urgent;
      const urgentIcon = urgentLabel.previousElementSibling;
      if (urgentIcon) urgentIcon.className = `fa-solid ${copy.icon}`;
    }
    if (btnJumpToWiki) btnJumpToWiki.classList.toggle('hidden', mode === 'retake');

    // Priority note: only for mixed diseases or an uncertain result. For a single
    // disease the backend note is just the lesion-area estimate, which we don't show.
    if (priorityDirectiveBox && guidePriorityNote) {
      const priorityText = userText(treatmentPlan.priority_note);
      const showPriority = !!priorityText && (conclusion.is_multi_disease || mode === 'refer');
      priorityDirectiveBox.classList.toggle('hidden', !showPriority);
      guidePriorityNote.textContent = showPriority ? priorityText : '';
    }

    // Garden care / next steps. The first action is already shown in the
    // "Nên làm ngay" box right above, so it is not repeated in the list.
    if (guideUrgentList) {
      guideUrgentList.innerHTML = '';
      const planActions = (treatmentPlan.urgent_actions || []).map(userText).filter(Boolean);
      let actions = planActions.length > 0
        ? planActions
        : ['Duy trì vệ sinh thông thoáng vườn điều và theo dõi định kỳ.'];
      const shownAbove = nextActionText ? nextActionText.textContent : '';
      const remaining = actions.filter(action => !shownAbove.includes(action));
      if (remaining.length > 0) actions = remaining;
      actions.forEach(action => {
        const li = document.createElement('li');
        li.textContent = action;
        guideUrgentList.appendChild(li);
      });
    }

    // Crop-protection products
    if (chemicalRecsWrap && guideChemicalBlock) {
      chemicalRecsWrap.innerHTML = '';
      const recs = treatmentPlan.chemical_recommendations || [];
      if (mode === 'refer') {
        guideChemicalBlock.classList.remove('hidden');
        chemicalRecsWrap.innerHTML = `
          <div class="chemical-item-card is-caution">
            <strong>Khuyến cáo an toàn:</strong> Tạm ngưng phun hóa chất bảo vệ thực vật diện rộng khi chưa có ý kiến của cán bộ chuyên môn / trạm BVTV địa phương.
          </div>
        `;
      } else if (recs.length > 0) {
        guideChemicalBlock.classList.remove('hidden');
        recs.forEach(chem => {
          const card = document.createElement('div');
          card.className = 'chemical-item-card';
          const actives = conclusion.label === 'healthy'
            ? escapeHtml(userText(chem.actives))
            : `<strong>Hoạt chất:</strong> ${escapeHtml(userText(chem.actives))}`;
          card.innerHTML = `
            <div class="chem-target"><i class="fa-solid fa-shield-virus"></i> ${escapeHtml(userText(chem.target))}</div>
            <div class="chem-actives">${actives}</div>
            <div class="chem-timing"><i class="fa-regular fa-clock"></i> ${escapeHtml(userText(chem.timing))}</div>
          `;
          chemicalRecsWrap.appendChild(card);
        });
      } else {
        guideChemicalBlock.classList.add('hidden');
      }
    }

    // Prevention
    if (guidePreventionList && guidePreventionBlock) {
      guidePreventionList.innerHTML = '';
      const items = (treatmentPlan.long_term_prevention || []).map(userText).filter(Boolean);
      guidePreventionBlock.classList.toggle('hidden', items.length === 0);
      items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        guidePreventionList.appendChild(li);
      });
    }
  }

  // ================= 4. FULL-SCREEN LAYERS: HISTORY, VIEWER, CAMERA =================

  const isTouchDevice = !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches);

  function lockScroll(locked) {
    document.documentElement.classList.toggle('overlay-open', locked);
  }

  /**
   * Each full-screen layer pushes one history entry, so the phone's Back button
   * (or browser Back) closes the layer instead of leaving the page.
   * Closing from the UI goes through history.back(), and 'popstate' does the closing.
   */
  function pushOverlay(name, closeFn) {
    const entry = { name, closeFn, hasHistory: false };
    try {
      history.pushState({ overlay: name }, '');
      entry.hasHistory = true;
    } catch (_) { /* history is not available (e.g. sandboxed frame) */ }
    overlayStack.push(entry);
  }

  function dismissOverlay(name) {
    const top = overlayStack[overlayStack.length - 1];
    if (!top || top.name !== name) return;
    if (top.hasHistory) {
      history.back();
    } else {
      overlayStack.pop();
      top.closeFn();
    }
  }

  window.addEventListener('popstate', () => {
    const top = overlayStack.pop();
    if (top) top.closeFn();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const top = overlayStack[overlayStack.length - 1];
    if (top) dismissOverlay(top.name);
  });

  // ---------- Full-screen image viewer ----------

  function openViewer({ src, title, withBoxes = false }) {
    if (!src || viewerOpen) return;

    viewerHasBoxes = !!(withBoxes && overlayState.meta && overlayState.boxes.length > 0);
    modalImgTitle.textContent = title || 'Chi Tiết Hình Ảnh';
    if (modalImgSub) {
      modalImgSub.textContent = viewerHasBoxes ? `${overlayState.boxes.length} vùng nghi bệnh được đánh dấu` : '';
      modalImgSub.classList.toggle('hidden', !viewerHasBoxes);
    }
    if (btnModalToggleBbox) {
      btnModalToggleBbox.classList.toggle('hidden', !viewerHasBoxes);
      btnModalToggleBbox.classList.toggle('active', isBboxVisible);
      btnModalToggleBbox.setAttribute('aria-pressed', String(isBboxVisible));
    }
    if (modalBboxLabel) modalBboxLabel.textContent = isBboxVisible ? 'Ẩn khung' : 'Hiện khung';

    viewerSvg.innerHTML = '';
    viewerSvg.style.display = isBboxVisible ? 'block' : 'none';
    modalFullImg.classList.remove('raw-orientation');
    modalFullImg.style.width = '';
    modalFullImg.style.height = '';

    viewerReturnFocus = document.activeElement;
    viewerOpen = true;
    imageModal.classList.remove('hidden');
    lockScroll(true);
    pushOverlay('viewer', closeViewer);

    modalFullImg.onload = () => layoutSurface(viewerSurface, true);
    modalFullImg.src = src;
    if (modalFullImg.complete && modalFullImg.naturalWidth > 0) {
      requestAnimationFrame(() => layoutSurface(viewerSurface, true));
    }
    btnCloseModal.focus({ preventScroll: true });
  }

  function closeViewer() {
    viewerOpen = false;
    viewerHasBoxes = false;
    imageModal.classList.add('hidden');
    viewerSvg.innerHTML = '';
    lockScroll(false);
    if (viewerReturnFocus && typeof viewerReturnFocus.focus === 'function') {
      try { viewerReturnFocus.focus({ preventScroll: true }); } catch (_) { /* element is gone */ }
    }
    viewerReturnFocus = null;
  }

  if (viewerStage && 'ResizeObserver' in window) {
    // Phone rotated / window resized while the viewer is open: refit the photo and redraw the boxes
    let viewerRaf = 0;
    new ResizeObserver(() => {
      if (!viewerOpen) return;
      cancelAnimationFrame(viewerRaf);
      viewerRaf = requestAnimationFrame(() => layoutSurface(viewerSurface, false));
    }).observe(viewerStage);
  }

  btnZoomImage.addEventListener('click', () => {
    if (viewerSource) openViewer({ src: viewerSource.src, title: viewerSource.title, withBoxes: true });
  });
  btnCloseModal.addEventListener('click', () => dismissOverlay('viewer'));
  // Boxes stop propagation, so this is a click on the empty picture area: leave single-box view
  viewerStage.addEventListener('click', clearIsolate);

  // ---------- Camera ----------

  let camStream = null;
  let camFacing = 'environment';
  let camBlob = null;
  let camSnapshotUrl = null;
  let camSession = 0;           // bumped to cancel a pending getUserMedia
  let camPausedByHide = false;
  let camReturnFocus = null;

  function canUseLiveCamera() {
    return !!(window.isSecureContext && navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function');
  }

  /** Opens the in-page camera, or the device's own camera app when the browser blocks live camera. */
  function openCamera() {
    if (!canUseLiveCamera()) {
      showToast(
        isTouchDevice
          ? 'Đang mở camera của thiết bị…'
          : 'Trình duyệt chưa cho dùng camera trực tiếp trên kết nối này — hãy chọn ảnh từ máy.',
        'info'
      );
      openNativeCamera();
      return;
    }
    if (!cameraModal.classList.contains('hidden')) return;

    camReturnFocus = document.activeElement;
    cameraModal.classList.remove('hidden');
    lockScroll(true);
    pushOverlay('camera', closeCameraLayer);
    startCamera();
  }

  function openNativeCamera() {
    cameraInput.value = '';
    cameraInput.click();
  }

  function stopCameraStream() {
    if (camStream) camStream.getTracks().forEach(track => track.stop());
    camStream = null;
    cameraVideo.srcObject = null;
  }

  function setCameraState(state, message) {
    cameraModal.dataset.state = state;
    cameraVideo.classList.toggle('hidden', state !== 'live');
    cameraSnapshot.classList.toggle('hidden', state !== 'review');
    cameraStatus.classList.toggle('hidden', state !== 'loading' && state !== 'error');
    cameraStatusIcon.className = state === 'error' ? 'fa-solid fa-video-slash' : 'fa-solid fa-circle-notch fa-spin';
    cameraStatusText.textContent = message || (state === 'loading' ? 'Đang mở camera…' : '');
    cameraStatusActions.classList.toggle('hidden', state !== 'error');
    cameraHint.classList.toggle('hidden', state !== 'live');
    btnCameraShutter.disabled = state !== 'live';
  }

  function cameraErrorMessage(err) {
    switch (err && err.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
      case 'SecurityError':
        return 'Trình duyệt chưa được cấp quyền dùng camera. Hãy cho phép camera cho trang này rồi thử lại.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
      case 'OverconstrainedError':
        return 'Không tìm thấy camera trên thiết bị này.';
      case 'NotReadableError':
      case 'TrackStartError':
      case 'AbortError':
        return 'Không mở được camera. Có thể ứng dụng khác đang dùng camera — hãy đóng nó rồi thử lại.';
      default:
        return 'Không mở được camera. Bạn có thể chọn ảnh từ thiết bị thay thế.';
    }
  }

  async function startCamera() {
    const session = ++camSession;
    stopCameraStream();
    setCameraState('loading');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: camFacing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      if (session !== camSession || cameraModal.classList.contains('hidden')) {
        stream.getTracks().forEach(track => track.stop());   // closed while the browser was asking permission
        return;
      }
      camStream = stream;
      cameraVideo.srcObject = stream;
      try { await cameraVideo.play(); } catch (_) { /* autoplay is allowed for muted inline video */ }

      const track = stream.getVideoTracks()[0];
      const settings = (track && track.getSettings && track.getSettings()) || {};
      cameraVideo.classList.toggle('is-mirrored', (settings.facingMode || camFacing) === 'user');
      setCameraState('live');
      updateCameraSwitch();
    } catch (err) {
      if (session !== camSession) return;
      console.warn('Camera error:', err);
      setCameraState('error', cameraErrorMessage(err));
    }
  }

  async function updateCameraSwitch() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter(d => d.kind === 'videoinput').length;
      btnCameraSwitch.classList.toggle('hidden', cameras < 2);
    } catch (_) {
      btnCameraSwitch.classList.add('hidden');
    }
  }

  function captureCameraFrame() {
    const vw = cameraVideo.videoWidth;
    const vh = cameraVideo.videoHeight;
    if (!vw || !vh) return;

    const k = Math.min(1, 2560 / Math.max(vw, vh));   // keep the upload reasonably small
    cameraCanvas.width = Math.round(vw * k);
    cameraCanvas.height = Math.round(vh * k);
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

    cameraCanvas.toBlob((blob) => {
      if (!blob) {
        showToast('Không chụp được ảnh, hãy thử lại.', 'error');
        return;
      }
      camBlob = blob;
      if (camSnapshotUrl) URL.revokeObjectURL(camSnapshotUrl);
      camSnapshotUrl = URL.createObjectURL(blob);
      cameraSnapshot.src = camSnapshotUrl;
      cameraVideo.pause();
      setCameraState('review');
    }, 'image/jpeg', 0.92);
  }

  function retakeCameraPhoto() {
    camBlob = null;
    if (camStream && camStream.active) {
      cameraVideo.play().catch(() => { });
      setCameraState('live');
    } else {
      startCamera();
    }
  }

  function useCameraPhoto() {
    if (!camBlob) return;
    const d = new Date();
    const pad = n => String(n).padStart(2, '0');
    const name = `camera_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.jpg`;
    const file = new File([camBlob], name, { type: 'image/jpeg', lastModified: Date.now() });
    dismissOverlay('camera');
    handleSelectedFiles([file]);
  }

  function closeCameraLayer() {
    camSession++;                 // cancels a pending getUserMedia
    stopCameraStream();
    camBlob = null;
    camPausedByHide = false;
    if (camSnapshotUrl) URL.revokeObjectURL(camSnapshotUrl);
    camSnapshotUrl = null;
    cameraSnapshot.removeAttribute('src');
    cameraModal.classList.add('hidden');
    lockScroll(false);
    if (camReturnFocus && typeof camReturnFocus.focus === 'function') {
      try { camReturnFocus.focus({ preventScroll: true }); } catch (_) { /* element is gone */ }
    }
    camReturnFocus = null;
  }

  // Release the camera while the tab is in the background, resume when it comes back
  document.addEventListener('visibilitychange', () => {
    if (cameraModal.classList.contains('hidden')) return;
    if (document.hidden) {
      if (cameraModal.dataset.state === 'live') {
        stopCameraStream();
        camPausedByHide = true;
      }
    } else if (camPausedByHide) {
      camPausedByHide = false;
      startCamera();
    }
  });

  if (btnOpenCamera) btnOpenCamera.addEventListener('click', openCamera);
  if (btnCameraResult) btnCameraResult.addEventListener('click', openCamera);
  btnCameraClose.addEventListener('click', () => dismissOverlay('camera'));
  btnCameraShutter.addEventListener('click', captureCameraFrame);
  btnCameraRetake.addEventListener('click', retakeCameraPhoto);
  btnCameraUse.addEventListener('click', useCameraPhoto);
  btnCameraRetry.addEventListener('click', startCamera);
  btnCameraFallback.addEventListener('click', openNativeCamera);
  btnCameraSwitch.addEventListener('click', () => {
    camFacing = camFacing === 'environment' ? 'user' : 'environment';
    startCamera();
  });

  // Photo taken with the device's own camera app (fallback path)
  cameraInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleSelectedFiles(files);
      dismissOverlay('camera');
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

    const icon = document.createElement('i');
    icon.className = `fa-solid ${iconClass}`;
    const text = document.createElement('span');
    text.textContent = message;
    toast.append(icon, text);

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
