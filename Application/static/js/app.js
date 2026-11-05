
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
  const conclusionLatin = document.getElementById('conclusion-latin');
  const conclusionAdvisoryText = document.getElementById('conclusion-advisory-text');
  const diseaseIconBox = document.getElementById('disease-icon-box');
  const conclusionIcon = document.getElementById('conclusion-icon');
  const detectedDiseasesRow = document.getElementById('detected-diseases-row');
  const severityBadgeTag = document.getElementById('severity-badge-tag');
  const severityStatText = document.getElementById('severity-stat-text');
  const severityProgressFill = document.getElementById('severity-progress-fill');

  // Lesion Breakdown elements
  const lesionTotalBadge = document.getElementById('lesion-total-badge');
  const lesionGridTable = document.getElementById('lesion-grid-table');

  // Treatment Protocol elements
  const guideProtocolHeading = document.getElementById('guide-protocol-heading');
  const priorityDirectiveBox = document.getElementById('priority-directive-box');
  const guidePriorityNote = document.getElementById('guide-priority-note');
  const guideUrgentList = document.getElementById('guide-urgent-list');
  const chemicalRecsWrap = document.getElementById('chemical-recs-wrap');
  const guidePreventionList = document.getElementById('guide-prevention-list');
  const btnJumpToWiki = document.getElementById('btn-jump-to-wiki');

  // Confidence List
  const confidenceList = document.getElementById('confidence-list');

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
  let lastDiagnosisResult = null;
  let currentLesionFilter = 'all';
  let isBboxVisible = true;

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
  if (btnToggleBbox && lesionSvgOverlay) {
    btnToggleBbox.addEventListener('click', () => {
      isBboxVisible = !isBboxVisible;
      lesionSvgOverlay.style.display = isBboxVisible ? 'block' : 'none';
      btnToggleBbox.classList.toggle('active', isBboxVisible);
      showToast(isBboxVisible ? 'Đã bật hiển thị khung khoanh vùng AI' : 'Đã ẩn khung khoanh vùng AI', 'info');
    });
  }

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
    lastDiagnosisResult = result;
    currentLesionFilter = 'all';

    emptyStateView.classList.add('hidden');
    activeResultView.classList.remove('hidden');
    setStatus('Đã chuẩn đoán');

    // 1. Image Preview & Dimensions
    imgFileName.textContent = fileName;
    imgFileSize.textContent = fileSizeStr;
    imageDimBadge.textContent = `${result.image_meta.width} × ${result.image_meta.height} px`;
    modalFullImg.src = imageUrl;
    if (modalImgTitle) {
      modalImgTitle.textContent = `Hình Ảnh Chuẩn Đoán: ${fileName}`;
    }

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
    //     input-quality notes are no longer displayed.
    renderInputWarnings(result);

    // 3. Render Lesion Filter Toolbar
    renderFilterChips(result.diseases_detected || [], result.total_lesion_boxes || 0, result.boxes || []);

    // 4. Primary Conclusion & Multi-Disease Mode
    const conclusion = result.conclusion;
    const conclusionCard = document.getElementById('conclusion-card');
    conclusionDiseaseName.textContent = conclusion.label_vn;
    conclusionLatin.textContent = conclusion.scientific_name || '';

    // Populate Indicator 1: Whole-Leaf Confidence
    const globalConfEl = document.getElementById('global-conf-score');
    if (globalConfEl) {
      globalConfEl.textContent = (conclusion.confidence_percent === null || conclusion.confidence_percent === undefined)
        ? '—'
        : `${conclusion.confidence_percent}%`;
    }

    if (conclusion.abstain) {
      if (conclusionCard) conclusionCard.classList.add('is-abstain');
      conclusionModeTag.innerHTML = '<i class="fa-solid fa-user-doctor"></i> CẦN CHUYÊN GIA KIỂM TRA';
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
      conclusionAdvisoryText.innerHTML = (conclusion.symptoms || '') + top2Html;
    } else {
      if (conclusionCard) conclusionCard.classList.remove('is-abstain');
      conclusionAdvisoryText.textContent = conclusion.symptoms || '';

      if (conclusion.is_multi_disease) {
        conclusionModeTag.innerHTML = '<i class="fa-solid fa-viruses"></i> ĐA BỆNH (MULTI-DISEASE)';
        conclusionModeTag.className = 'badge-conclusion-tag multi-mode';
      } else if (conclusion.label === 'healthy') {
        conclusionModeTag.innerHTML = '<i class="fa-solid fa-shield-halved"></i> LÁ KHỎE MẠNH';
        conclusionModeTag.className = 'badge-conclusion-tag';
      } else if (conclusion.label === 'not_cashew_leaf') {
        conclusionModeTag.innerHTML = conclusion.ood_rejected
          ? '<i class="fa-solid fa-ban"></i> TỪ CHỐI — ẢNH NGOẠI LAI'
          : '<i class="fa-solid fa-triangle-exclamation"></i> NGOẠI LAI (CẢNH BÁO)';
        conclusionModeTag.className = 'badge-conclusion-tag';
      } else {
        conclusionModeTag.innerHTML = '<i class="fa-solid fa-check"></i> ĐƠN BỆNH (SINGLE)';
        conclusionModeTag.className = 'badge-conclusion-tag';
      }
    }

    diseaseIconBox.style.color = conclusion.color_hex;
    diseaseIconBox.style.borderColor = conclusion.color_hex;
    conclusionIcon.className = `fa-solid ${conclusion.icon}`;

    // 5. Render Detected Diseases Row
    renderDetectedDiseasesRow(result.diseases_detected || []);
    renderVerificationSummary(result);

    // 6. Severity Assessment Meter
    const totalArea = conclusion.total_damage_area_percent || 0;
    const totalBoxes = result.total_lesion_boxes || 0;
    const sevLevel = conclusion.severity_level || 'low';

    let sevBadgeText = 'Nhẹ — Cấp độ 1';
    let sevBadgeClass = 'sev-low';
    let sevBarColor = '#15803d';

    if (conclusion.abstain || sevLevel === 'abstain') {
      sevBadgeText = 'Cần Thẩm Định';
      sevBadgeClass = 'sev-abstain';
      sevBarColor = '#eab308';
    } else if (sevLevel === 'high') {
      sevBadgeText = 'Nặng — Cấp độ 3';
      sevBadgeClass = 'sev-high';
      sevBarColor = '#dc2626';
    } else if (sevLevel === 'medium') {
      sevBadgeText = 'Trung bình — Cấp độ 2';
      sevBadgeClass = 'sev-medium';
      sevBarColor = '#b45309';
    } else if (sevLevel === 'healthy') {
      sevBadgeText = 'An toàn';
      sevBadgeClass = 'sev-low';
      sevBarColor = '#10b981';
    }

    severityBadgeTag.textContent = sevBadgeText;
    severityBadgeTag.className = `severity-badge-tag ${sevBadgeClass}`;
    const areaBasis = conclusion.damage_basis === 'image' ? 'diện tích ảnh' : 'diện tích lá';
    severityStatText.textContent = `≤ ${totalArea}% ${areaBasis} (${totalBoxes} vùng khoanh, ước tính theo khung)`;
    severityProgressFill.style.width = `${Math.min(100, Math.max(totalArea, 4))}%`;
    severityProgressFill.style.background = sevBarColor;

    // 7. Render Lesion Breakdown Table & Thumbnails
    renderLesionBreakdown(result.boxes || []);

    // 8. Render Actionable Treatment Protocol & IPM
    renderTreatmentProtocol(result.treatment_plan, conclusion);

    // 9. Render Global DenseNet Confidence Breakdown
    const listDesc = result.probabilities_desc || [];
    renderDescendingConfidenceList(listDesc, conclusion.label);

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
    const img = diagnosisPreviewImg;
    const canvasBox = document.getElementById('preview-canvas-box');
    const meta = overlayState.meta;
    if (!img || !lesionSvgOverlay || !meta) return;

    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    if (!nw || !nh) return;

    const mw = meta.width || nw;
    const mh = meta.height || nh;

    if (isNewImage && !img.classList.contains('raw-orientation') &&
        nw !== nh && nw === mh && nh === mw) {
      img.classList.add('raw-orientation');
      requestAnimationFrame(() => syncOverlayToImage(false));
      return;
    }

    // Fit the image to the viewing frame (contain). Small photos are UPSCALED
    // so lesions are visible; the canvas box follows the <img> size exactly,
    // so the SVG overlay stays aligned 1:1 with the image pixels.
    if (previewWrapper) {
      const info = document.getElementById('img-overlay-info');
      const availW = previewWrapper.clientWidth;
      const availH = previewWrapper.clientHeight - (info ? info.offsetHeight : 0);
      if (availW > 0 && availH > 0) {
        const fit = Math.min(availW / nw, availH / nh);
        img.style.width = `${Math.floor(nw * fit)}px`;
        img.style.height = `${Math.floor(nh * fit)}px`;
      }
    }

    // Since .preview-canvas-box wraps <img> directly, SVG overlay (width: 100%, height: 100%)
    // has the EXACT same rendered dimensions and aspect ratio as the visible image.
    // SVG viewBox defines the original coordinate space from the server:
    lesionSvgOverlay.setAttribute('viewBox', `0 0 ${mw} ${mh}`);
    lesionSvgOverlay.setAttribute('preserveAspectRatio', 'none');

    const renderedW = img.clientWidth || (canvasBox ? canvasBox.clientWidth : nw);
    const zoom = computeZoom(canvasBox, mw);
    // Divide by the zoom so labels keep a constant on-screen size when zoomed
    const unitsPerPx = renderedW > 0 ? (mw / renderedW) / zoom.s : 1;

    drawLesionBoxes(overlayState.boxes, mw, mh, unitsPerPx);
    if (canvasBox) {
      canvasBox.style.transform = zoom.s === 1 ? '' : `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.s})`;
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
  function drawLesionBoxes(boxes, frameW, frameH, unitsPerPx) {
    if (!lesionSvgOverlay) return;
    lesionSvgOverlay.innerHTML = '';
    if (!boxes || boxes.length === 0) return;

    const k = unitsPerPx > 0 ? unitsPerPx : 1;
    const fontSize = 11 * k;
    const tagH = 18 * k;
    const padX = 5 * k;
    const svgNS = 'http://www.w3.org/2000/svg';

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

      // Label Tag Text: Show Spot ID, Disease Name, and Prediction Confidence %
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('class', 'lesion-tag-text');
      text.setAttribute('font-size', fontSize);
      const differ = box.verify && box.verify.status === 'differ';
      text.textContent = (box.conf_percent === null || box.conf_percent === undefined)
        ? `#${box.id} ${box.label_vn}`
        : `#${box.id} ${box.label_vn} ${box.conf_percent}%${differ ? ' ≠YOLO' : ''}`;

      const tagBg = document.createElementNS(svgNS, 'rect');
      tagBg.setAttribute('class', 'lesion-tag-bg');
      tagBg.setAttribute('rx', 2 * k);
      tagBg.setAttribute('ry', 2 * k);
      tagBg.setAttribute('height', tagH);
      tagBg.setAttribute('fill', box.color_hex);

      g.appendChild(rect);
      g.appendChild(tagBg);
      g.appendChild(text);
      lesionSvgOverlay.appendChild(g);

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

      if (overlayState.focusedId === box.id) g.classList.add('is-focused');

      // Click on a box -> show ONLY that box (click again to show all)
      g.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleIsolate(box.id, false);
      });
    });

    applyLesionVisibility();
    const focused = lesionSvgOverlay.querySelector('.lesion-box-group.is-focused');
    if (focused) lesionSvgOverlay.appendChild(focused);
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
        : `<i class="fa-solid fa-viruses"></i> Phát hiện <strong>${groups.length} loại bệnh</strong> · <strong>${boxes.length} vùng</strong>
           <span class="vs-ref">— chọn một bệnh để lọc, bấm vào một khung hoặc thẻ để chỉ xem vùng đó</span>`;
    }

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

    if (lesionSvgOverlay) {
      lesionSvgOverlay.querySelectorAll('.lesion-box-group').forEach(g => {
        const id = parseInt(g.getAttribute('data-box-id'), 10);
        g.classList.toggle('is-hidden', !visible(g.getAttribute('data-cls'), id));
        g.classList.toggle('is-focused', iso !== null && id === iso);
      });
    }
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
  function toggleIsolate(boxId, fromCard) {
    overlayState.isolatedId = overlayState.isolatedId === boxId ? null : boxId;
    overlayState.focusedId = overlayState.isolatedId;
    applyLesionVisibility();
    applyZoom();
    if (overlayState.isolatedId === null) return;
    if (fromCard && previewWrapper) {
      previewWrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (lesionGridTable) {
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
    if (e.key === 'Escape' && imageModal && imageModal.classList.contains('hidden')) clearIsolate();
  });

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
    // Keep the warning INSIDE the conclusion card: inserting it as a sibling
    // made it a cell of the 2-column .diagnosis-top-grid and pushed the
    // conclusion card into the narrow column of the next row.
    if (box.parentNode !== card) card.insertBefore(box, card.firstChild);
    const items = [];
    const engineWarning = result.model_meta && result.model_meta.engine_warning;
    if (engineWarning) items.push({ icon: 'fa-server', text: engineWarning, strong: true });
    if (items.length === 0) {
      box.classList.add('hidden');
      box.innerHTML = '';
      return;
    }
    box.classList.remove('hidden');
    box.innerHTML = `
      <div class="input-warning-title"><i class="fa-solid fa-triangle-exclamation"></i> Cảnh báo hệ thống</div>
      <ul>${items.map(i => `<li class="${i.strong ? 'is-strong' : ''}"><i class="fa-solid ${i.icon}"></i> ${escapeHtml(i.text)}</li>`).join('')}</ul>
    `;
  }

  function renderVerificationSummary(result) {
    if (!detectedDiseasesRow || !detectedDiseasesRow.parentNode) return;
    let box = document.getElementById('verification-summary-box');
    if (!box) {
      box = document.createElement('div');
      box.id = 'verification-summary-box';
      box.className = 'verification-summary-box';
      detectedDiseasesRow.parentNode.insertBefore(box, detectedDiseasesRow.nextSibling);
    }
    const dec = result.decision;
    const loc = result.localization || {};
    const v = loc.verification || {};
    if (!dec) { box.classList.add('hidden'); box.innerHTML = ''; return; }
    box.classList.remove('hidden');

    const gate = result.ood_gate;
    if (dec.final_source === 'ood_gate' && gate) {
      const fmt = (v, d) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);
      box.innerHTML = `
        <div class="vs-title"><i class="fa-solid fa-ban"></i> Bước 0 — Cổng kiểm tra ngoại lai: ẢNH BỊ TỪ CHỐI</div>
        <ul class="vs-reasons">${gate.reasons.map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul>
        <div class="vs-chips">
          <span class="vs-chip muted">P(không phải lá điều) ${(gate.p_not_cashew * 100).toFixed(1)}%</span>
          <span class="vs-chip muted">kNN ${fmt(gate.knn_distance, 3)} / vùng nghi ngờ ${fmt(gate.knn_threshold, 3)}–${fmt(gate.knn_threshold_high, 3)}</span>
          <span class="vs-chip muted">Energy ${fmt(gate.energy, 2)}${gate.energy_used ? ` / ngưỡng ${fmt(gate.energy_threshold, 2)}` : ' (tham khảo)'}</span>
        </div>
        <div class="vs-row"><i class="fa-solid fa-lock"></i> Luật cứng: YOLO không được chạy, không khoanh vùng và không chẩn đoán bệnh.</div>`;
      return;
    }

    const name = key => {
      const it = (result.probabilities_desc || []).find(p => p.class_key === key);
      return it ? (it.short_name || it.label_vn) : key;
    };
    const finalSrc = dec.final_source === 'densenet' ? 'DenseNet121 ×5 (toàn lá)' : 'Stacker F3';
    const other = dec.final_source === 'densenet'
      ? `Stacker F3: ${name(dec.stacker_label)} ${(dec.stacker_conf * 100).toFixed(1)}%`
      : `DenseNet: ${name(dec.densenet_label)} ${(dec.densenet_conf * 100).toFixed(1)}%`;

    let verifyLine = '';
    if (v.proposed !== undefined && v.agree !== undefined) {
      verifyLine = `
        <div class="vs-row"><i class="fa-solid fa-crosshairs"></i> YOLO đề xuất <strong>${v.proposed}</strong> vùng
          → DenseNet xác minh qua <strong>${loc.verify_crops || 0}</strong> vùng cắt:</div>
        <div class="vs-chips">
          <span class="vs-chip ok"><i class="fa-solid fa-check"></i> Khớp ${v.agree}</span>
          <span class="vs-chip warn"><i class="fa-solid fa-not-equal"></i> Khác nhãn ${v.differ}</span>
          <span class="vs-chip muted"><i class="fa-solid fa-leaf"></i> DenseNet bác bỏ (lá khỏe) ${v.rejected_healthy}</span>
          <span class="vs-chip muted"><i class="fa-solid fa-eye-slash"></i> Ẩn (bệnh khác chưa đủ bằng chứng) ${v.hidden_other_disease}</span>
          ${v.unverified ? `<span class="vs-chip muted">Chưa xác minh ${v.unverified}</span>` : ''}
        </div>`;
    } else if (v.proposed !== undefined) {
      verifyLine = `<div class="vs-row"><i class="fa-solid fa-crosshairs"></i> YOLO đề xuất ${v.proposed} vùng; DenseNet toàn lá kết luận không bệnh nên không khoanh.</div>`;
    }

    let gateLine = '';
    if (gate && !gate.calibrated) {
      gateLine = `<div class="vs-row"><i class="fa-solid fa-shield-halved"></i> Bước 0 — Cổng ngoại lai: <strong>đạt</strong>
        <span class="vs-ref is-conflict">(chưa hiệu chỉnh kNN — chỉ dùng lớp not_cashew_leaf)</span></div>`;
    } else if (gate && gate.zone === 'band') {
      const top = gate.band_rule ? (gate.band_rule.top_cashew * 100).toFixed(1) : '—';
      gateLine = `<div class="vs-row"><i class="fa-solid fa-shield-halved"></i> Bước 0 — Cổng ngoại lai:
        <strong class="vs-ref is-conflict">qua cổng có điều kiện</strong>
        (kNN ${Number(gate.knn_distance).toFixed(3)} trong vùng nghi ngờ ${Number(gate.knn_threshold).toFixed(3)}–${Number(gate.knn_threshold_high).toFixed(3)};
        được chấp nhận vì DenseNet chắc là lá điều: ${top}%, P(không phải lá điều) ${(gate.p_not_cashew * 100).toFixed(1)}%). Nên kiểm tra lại bằng ảnh rõ hơn.</div>`;
    } else if (gate) {
      gateLine = `<div class="vs-row"><i class="fa-solid fa-shield-halved"></i> Bước 0 — Cổng ngoại lai: <strong>đạt</strong>
        (kNN ${Number(gate.knn_distance).toFixed(3)} ≤ ${Number(gate.knn_threshold).toFixed(3)}; Energy ${Number(gate.energy).toFixed(2)} — chỉ tham khảo)</div>`;
    }
    box.innerHTML = `
      <div class="vs-title"><i class="fa-solid fa-diagram-project"></i> Quy trình: Cổng ngoại lai → YOLO khoanh vùng → DenseNet xác minh &amp; kết luận</div>
      ${gateLine}
      <div class="vs-row"><i class="fa-solid fa-brain"></i> Kết luận cuối: <strong>${finalSrc}</strong>
        <span class="vs-ref ${dec.agree ? '' : 'is-conflict'}">(đối chiếu ${escapeHtml(other)}${dec.agree ? ' — trùng' : ' — KHÁC'})</span></div>
      ${verifyLine}
    `;
  }

  function renderDetectedDiseasesRow(diseasesDetected) {
    if (!detectedDiseasesRow) return;
    detectedDiseasesRow.innerHTML = '';

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
      lesionTotalBadge.textContent = `${boxes.length} vùng bệnh`;
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
          ${box.verify ? (box.verify.status === 'unverified' ? `
          <div class="lesion-verify-row is-unverified">
            <span title="Nhãn YOLO đề xuất cho khung này"><i class="fa-solid fa-crosshairs"></i> YOLO: <strong>${box.verify.yolo_label_vn} ${box.verify.yolo_conf_percent}%</strong></span>
            <span class="verify-badge" title="DenseNet không đọc được vùng cắt (xác suất 3 bệnh < 50%), giữ nhãn YOLO"><i class="fa-solid fa-circle-question"></i> Chưa xác minh</span>
          </div>` : `
          <div class="lesion-verify-row ${box.verify.status === 'agree' ? 'is-agree' : 'is-differ'}">
            <span title="DenseNet phân loại vùng cắt quanh khung (xác suất thô)"><i class="fa-solid fa-brain"></i> DenseNet: <strong>${box.label_vn} ${box.verify.dense_conf_percent}%</strong></span>
            <span title="Nhãn YOLO đề xuất cho khung này"><i class="fa-solid fa-crosshairs"></i> YOLO: ${box.verify.yolo_label_vn} ${box.verify.yolo_conf_percent}%</span>
            <span class="verify-badge">${box.verify.status === 'agree' ? '<i class="fa-solid fa-check"></i> Khớp' : '<i class="fa-solid fa-not-equal"></i> Khác nhãn'}</span>
          </div>`) : ''}
          <div class="lesion-card-bottom">
            ${(box.conf_percent === null || box.conf_percent === undefined)
              ? `<span class="lesion-conf-score" title="Vùng tìm theo màu sắc, không có điểm tin cậy của mô hình"><i class="fa-solid fa-circle-question"></i> Vùng nghi ngờ (chưa chấm điểm)</span>`
              : box.verify
                ? `<span class="lesion-conf-score"><i class="fa-solid fa-layer-group"></i> Vùng cắt #${(box.verify.crop_id ?? 0) + 1}</span>`
                : `<span class="lesion-conf-score"><i class="fa-solid fa-bullseye"></i> Điểm: <strong>${box.conf_percent}%</strong></span>`}
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

  function renderTreatmentProtocol(treatmentPlan, conclusion) {
    if (!treatmentPlan) return;

    if (guideProtocolHeading) {
      guideProtocolHeading.textContent = treatmentPlan.title || 'Phác Đồ Xử Lý & Điều Trị Khuyến Nghị';
    }

    // Priority note
    if (priorityDirectiveBox && guidePriorityNote) {
      if (treatmentPlan.priority_note) {
        priorityDirectiveBox.classList.remove('hidden');
        guidePriorityNote.textContent = treatmentPlan.priority_note;
      } else {
        priorityDirectiveBox.classList.add('hidden');
      }
    }

    // Urgent Cultivation Actions
    if (guideUrgentList) {
      guideUrgentList.innerHTML = '';
      if (treatmentPlan.urgent_actions && treatmentPlan.urgent_actions.length > 0) {
        treatmentPlan.urgent_actions.forEach(action => {
          const li = document.createElement('li');
          li.textContent = action;
          guideUrgentList.appendChild(li);
        });
      } else {
        const li = document.createElement('li');
        li.textContent = 'Duy trì vệ sinh thông thoáng vườn điều và theo dõi định kỳ.';
        guideUrgentList.appendChild(li);
      }
    }

    // Chemical Recommendations
    if (chemicalRecsWrap) {
      chemicalRecsWrap.innerHTML = '';
      if (conclusion && conclusion.abstain) {
        chemicalRecsWrap.innerHTML = `
          <div class="chemical-item-card" style="color: #854d0e; background: #fefce8; border-color: #facc15; font-size: 0.8rem;">
            <i class="fa-solid fa-triangle-exclamation" style="margin-right: 0.35rem; color: #eab308;"></i>
            <strong>Khuyến cáo an toàn:</strong> Tạm ngưng phun hóa chất bảo vệ thực vật diện rộng khi chưa có ý kiến của cán bộ chuyên môn / trạm BVTV địa phương.
          </div>
        `;
      } else if (treatmentPlan.chemical_recommendations && treatmentPlan.chemical_recommendations.length > 0) {
        treatmentPlan.chemical_recommendations.forEach(chem => {
          const card = document.createElement('div');
          card.className = 'chemical-item-card';
          card.innerHTML = `
            <div class="chem-target"><i class="fa-solid fa-shield-virus"></i> ${chem.target}</div>
            <div class="chem-actives"><strong>Hoạt chất:</strong> ${chem.actives}</div>
            <div class="chem-timing"><i class="fa-regular fa-clock"></i> ${chem.timing}</div>
          `;
          chemicalRecsWrap.appendChild(card);
        });
      } else {
        chemicalRecsWrap.innerHTML = `
          <div class="chemical-item-card" style="color: var(--text-muted); font-size: 0.78rem;">
            Không cần phun hóa chất bảo vệ thực vật. Khuyến khích sử dụng chế phẩm sinh học phòng ngừa.
          </div>
        `;
      }
    }

    // Prevention List
    if (guidePreventionList) {
      guidePreventionList.innerHTML = '';
      if (treatmentPlan.long_term_prevention && treatmentPlan.long_term_prevention.length > 0) {
        treatmentPlan.long_term_prevention.forEach(item => {
          const li = document.createElement('li');
          li.textContent = item;
          guidePreventionList.appendChild(li);
        });
      }
    }
  }

  function renderDescendingConfidenceList(classesDesc, winnerKey) {
    if (!confidenceList) return;
    confidenceList.innerHTML = '';

    classesDesc.forEach((item, index) => {
      const isWinner = item.class_key === winnerKey;
      const rankNumber = index + 1;

      const row = document.createElement('div');
      row.className = `confidence-row ${isWinner ? 'is-winner' : ''}`;

      row.innerHTML = `
        <div class="row-top-info">
          <div class="row-label-group">
            <span class="rank-index">#${rankNumber}</span>
            <span class="disease-vn-label">${item.label_vn}</span>
            <span class="disease-en-sub">(${item.class_key})</span>
            ${isWinner ? '<span class="winner-pill-tag"><i class="fa-solid fa-check"></i> XÁC SUẤT CAO NHẤT</span>' : ''}
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
