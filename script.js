
  async function testCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach(track => track.stop());
      alert('Camera access successful! Now click Start.');
    } catch (error) {
      alert('Camera error: ' + error.message);
    }
  }


    let video, canvas, ctx;
    let isRunning = false;
    let baselineFrames = [];
    let baselinePower = 0;
    let baselineEstablished = false;
    let currentIntensity = 0;
    let detectionCount = 0;
    let frameCount = 0;
    let intensityHistory = [];
    let graphCtx;
    let deferredPrompt;
    let lastFrameTime = 0;
    let fps = 0;
    let torchOn = false;
    
    // Constants matching Flask
    const BASELINE_FRAMES = 10;
    const DETECTION_THRESHOLD = 30.0;
    const ROI_WIDTH = 160;
    const ROI_HEIGHT = 120;
    const MAX_HISTORY = 60;
    const CLAHE_CLIP = 2.0;
    const FREQ_MIN_MM = 0.23;
    const FREQ_MAX_MM = 0.75;
    const PIXEL_SIZE_MM = 0.25;
    
    // ============================================
    // PWA Install Handler
    // ============================================
    
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      document.getElementById('installBanner').classList.add('show');
      document.getElementById('debugPwa').textContent = 'Ready';
    });
    
    document.getElementById('installBtn').addEventListener('click', async () => {
      if (!deferredPrompt) {
        alert('Install option not available. Try:\n1. Chrome menu → "Install app"\n2. Or "Add to Home Screen"');
        return;
      }
      
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      console.log('Install outcome:', outcome);
      deferredPrompt = null;
      document.getElementById('installBanner').classList.remove('show');
    });
    
    window.addEventListener('appinstalled', () => {
      console.log('✓ CHILLER installed!');
      document.getElementById('debugPwa').textContent = 'Installed';
    });
    
    // ============================================
    // FFT Implementation
    // ============================================
    
    function fft(real, imag) {
      const n = real.length;
      if (n <= 1) return;
      if (n % 2 !== 0) return;
      
      const evenReal = [], evenImag = [];
      const oddReal = [], oddImag = [];
      
      for (let i = 0; i < n / 2; i++) {
        evenReal[i] = real[2 * i];
        evenImag[i] = imag[2 * i];
        oddReal[i] = real[2 * i + 1];
        oddImag[i] = imag[2 * i + 1];
      }
      
      fft(evenReal, evenImag);
      fft(oddReal, oddImag);
      
      for (let k = 0; k < n / 2; k++) {
        const angle = -2 * Math.PI * k / n;
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);
        
        const tReal = cosA * oddReal[k] - sinA * oddImag[k];
        const tImag = cosA * oddImag[k] + sinA * oddReal[k];
        
        real[k] = evenReal[k] + tReal;
        imag[k] = evenImag[k] + tImag;
        real[k + n / 2] = evenReal[k] - tReal;
        imag[k + n / 2] = evenImag[k] - tImag;
      }
    }
    
    // ============================================
    // CHILLER Algorithm
    // ============================================
    
    function calculateGoosebumpPower(grayData, width, height) {
      let sum = 0, sumSq = 0;
      const n = grayData.length;
      
      for (let i = 0; i < n; i++) {
        sum += grayData[i];
        sumSq += grayData[i] * grayData[i];
      }
      
      const mean = sum / n;
      const variance = (sumSq / n) - (mean * mean);
      const std = Math.sqrt(variance);
      
      if (std < 1.0) return 0.0;
      
      const normalized = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        normalized[i] = (grayData[i] - mean) / std;
      }
      
      const powerSpectrum = new Float32Array(width);
      powerSpectrum.fill(0);
      
      for (let y = 0; y < height; y++) {
        const rowReal = new Float32Array(width);
        const rowImag = new Float32Array(width);
        
        for (let x = 0; x < width; x++) {
          rowReal[x] = normalized[y * width + x];
          rowImag[x] = 0;
        }
        
        fft(rowReal, rowImag);
        
        for (let x = 0; x < width; x++) {
          const power = rowReal[x] * rowReal[x] + rowImag[x] * rowImag[x];
          powerSpectrum[x] += power;
        }
      }
      
      for (let x = 0; x < width; x++) {
        powerSpectrum[x] /= height;
      }
      
      const nyquistFreq = 0.5;
      const freqMin = FREQ_MIN_MM * PIXEL_SIZE_MM;
      const freqMax = FREQ_MAX_MM * PIXEL_SIZE_MM;
      
      const binMin = Math.floor(freqMin / nyquistFreq * width / 2);
      const binMax = Math.ceil(freqMax / nyquistFreq * width / 2);
      
      let maxPower = 0;
      for (let i = binMin; i < Math.min(binMax, width / 2); i++) {
        if (powerSpectrum[i] > maxPower) {
          maxPower = powerSpectrum[i];
        }
      }
      
      return maxPower;
    }
    
    // ============================================
    // CLAHE Enhancement
    // ============================================
    
    function applyCLAHE(grayData, width, height) {
      const tileSize = 8;
      const tilesX = Math.floor(width / tileSize);
      const tilesY = Math.floor(height / tileSize);
      const enhanced = new Uint8Array(grayData.length);
      
      for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
          const hist = new Array(256).fill(0);
          
          for (let y = 0; y < tileSize; y++) {
            for (let x = 0; x < tileSize; x++) {
              const px = tx * tileSize + x;
              const py = ty * tileSize + y;
              if (px < width && py < height) {
                const idx = py * width + px;
                hist[grayData[idx]]++;
              }
            }
          }
          
          const clipLimit = (tileSize * tileSize * CLAHE_CLIP) / 256;
          let excess = 0;
          for (let i = 0; i < 256; i++) {
            if (hist[i] > clipLimit) {
              excess += hist[i] - clipLimit;
              hist[i] = clipLimit;
            }
          }
          const redistribute = excess / 256;
          for (let i = 0; i < 256; i++) {
            hist[i] += redistribute;
          }
          
          const cdf = new Array(256);
          cdf[0] = hist[0];
          for (let i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + hist[i];
          }
          
          const total = cdf[255];
          for (let i = 0; i < 256; i++) {
            cdf[i] = (cdf[i] / total) * 255;
          }
          
          for (let y = 0; y < tileSize; y++) {
            for (let x = 0; x < tileSize; x++) {
              const px = tx * tileSize + x;
              const py = ty * tileSize + y;
              if (px < width && py < height) {
                const idx = py * width + px;
                enhanced[idx] = Math.round(cdf[grayData[idx]]);
              }
            }
          }
        }
      }
      
      return enhanced;
    }
    
    // ============================================
    // Initialize
    // ============================================
    
    window.addEventListener('load', () => {
      setTimeout(() => {
        document.getElementById('splashScreen').classList.add('hidden');
      }, 2000);
      
      video = document.getElementById('videoElement');
      canvas = document.getElementById('processingCanvas');
      ctx = canvas.getContext('2d', { 
        willReadFrequently: true,
        alpha: false,
        desynchronized: true
      });
      
      const graphCanvas = document.getElementById('graphCanvas');
      graphCtx = graphCanvas.getContext('2d');
      graphCanvas.width = graphCanvas.offsetWidth;
      graphCanvas.height = 200;
      
      document.getElementById('startBtn').addEventListener('click', startCamera);
      document.getElementById('stopBtn').addEventListener('click', stopCamera);
      
      // Check if already installed
      if (window.matchMedia('(display-mode: standalone)').matches) {
        document.getElementById('debugPwa').textContent = 'Running';
      }
    });
    
    // ============================================
    // HIGH QUALITY Camera Setup
    // ============================================
    
    async function pickRearCameraDeviceId() {
  // Ensure we have permission so labels are available
  try { await navigator.mediaDevices.getUserMedia({ video: true }); } catch {}
  const devices = await navigator.mediaDevices.enumerateDevices();
  // Prefer labels indicating back/rear, else last video input (often rear on phones)
  const videoInputs = devices.filter(d => d.kind === 'videoinput');
  const rear = videoInputs.find(d =>
    /back|rear|environment/i.test(d.label || '')
  );
  return (rear || videoInputs[videoInputs.length - 1] || {}).deviceId;
}

async function startCamera() {
  const statusBanner = document.getElementById('statusBanner');
  try {
    statusBanner.textContent = 'Requesting high-quality camera…';

    if (!window.isSecureContext) {
      throw new Error('Page must be loaded over HTTPS. Current URL: ' + window.location.href);
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera API not supported in this browser');
    }

    // iOS: allow inline playback
    video.setAttribute('playsinline', '');
    video.setAttribute('autoplay', '');
    video.muted = true; // helps autoplay on some browsers

    const deviceId = await pickRearCameraDeviceId();

    // Try a resolution "ladder": 4K → 1080p → 720p
    const ladders = [
      { width: { ideal: 3840 }, height: { ideal: 2160 } }, // 4K
      { width: { ideal: 1920 }, height: { ideal: 1080 } }, // 1080p
      { width: { ideal: 1280 }, height: { ideal: 720 } },  // 720p
    ];

    let stream;
    let lastError;

    for (const res of ladders) {
      const constraints = {
        video: {
          ...res,
          deviceId: deviceId ? { exact: deviceId } : undefined,
          facingMode: 'environment',         // hint; deviceId is primary selector
          frameRate: { ideal: 30, max: 60 }  // smoother frames if possible
        },
        audio: false
      };
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        break; // success at this rung
      } catch (e) {
        lastError = e;
      }
    }

    if (!stream) throw lastError || new Error('Failed to get camera stream');

    // Bind and play
    video.srcObject = stream;
    await video.play().catch(() => {}); // some browsers need a user gesture anyway

    // Try upgrading the active track with capabilities (focus/exposure/white balance, etc.)
    const track = stream.getVideoTracks()[0];
    const caps = track.getCapabilities?.() || {};
    const advanced = [];

    // Continuous focus (Chromium / some Androids)
    if (caps.focusMode && caps.focusMode.includes('continuous')) {
      advanced.push({ focusMode: 'continuous' });
    }
    // Auto exposure / white balance where available
    if (caps.exposureMode && caps.exposureMode.includes('continuous')) {
      advanced.push({ exposureMode: 'continuous' });
    }
    if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes('continuous')) {
      advanced.push({ whiteBalanceMode: 'continuous' });
    }
    // Prefer 30–60 fps if supported
    if (caps.frameRate) {
      advanced.push({ frameRate: Math.min(60, caps.frameRate.max || 30) });
    }
    // Nudge up resolution to capability max if the browser lowballed us
    if (caps.width && caps.height) {
      advanced.push({
        width: Math.min(3840, caps.width.max || 1920),
        height: Math.min(2160, caps.height.max || 1080),
      });
    }

    if (advanced.length) {
      try { await track.applyConstraints({ advanced }); } catch (e) { /* ignore if unsupported */ }
    }

    // Update UI and canvas sizing
    isRunning = true;
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';
    document.getElementById('instructions').style.display = 'none';

    // Wait for metadata to ensure correct dimensions
    await new Promise(r => {
      if (video.readyState >= 1) return r();
      video.onloadedmetadata = () => r();
      setTimeout(r, 1000);
    });

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    console.log(`Actual stream:`, stream.getVideoTracks()[0].getSettings());
    statusBanner.textContent = `Camera: ${video.videoWidth}×${video.videoHeight}`;

    // Start processing
    processFrames();

    // Detect torch capability and show flashlight button when available
    try {
      const track = stream.getVideoTracks()[0];
      const caps = track.getCapabilities ? track.getCapabilities() : {};
      let hasTorch = false;
      if ('torch' in caps) {
        hasTorch = true;
      } else if (window.ImageCapture) {
        try {
          const ic = new ImageCapture(track);
          const pc = await ic.getPhotoCapabilities();
          if (pc && pc.fillLightMode && pc.fillLightMode.includes('torch')) hasTorch = true;
        } catch (e) {
          // ignore
        }
      }
      if (hasTorch) {
        const btn = document.getElementById('flashBtn');
        if (btn) btn.style.display = 'inline-block';
      }
    } catch (e) {
      console.log('Torch detection failed', e);
    }

  } catch (error) {
    console.error('Camera setup error:', error);
    const errorMsg =
      (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError')
        ? 'Camera access denied. Enable camera in site settings and refresh.'
        : `Camera error: ${error.message}\n\nCheck:\n1) HTTPS\n2) Permissions\n3) No other app using camera`;
    alert(errorMsg);
    statusBanner.textContent = 'Camera error - check permissions';
    statusBanner.style.backgroundColor = 'rgba(255,0,0,0.8)';
  }
}




    function stopCamera() {
      isRunning = false;
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
      }
      document.getElementById('startBtn').style.display = 'block';
      document.getElementById('stopBtn').style.display = 'none';
      resetState();
    }
    
    function resetState() {
      baselineFrames = [];
      baselinePower = 0;
      baselineEstablished = false;
      currentIntensity = 0;
      frameCount = 0;
      detectionCount = 0;
      intensityHistory = [];
      updateUI();
    }

    // ============================================
    // Flashlight / Torch Controls
    // ============================================

    async function setTorch(on) {
      try {
        if (!video || !video.srcObject) return false;
        const track = video.srcObject.getVideoTracks()[0];
        if (!track) return false;

        const caps = track.getCapabilities ? track.getCapabilities() : {};
        if ('torch' in caps) {
          try {
            await track.applyConstraints({ advanced: [{ torch: on }] });
            torchOn = !!on;
            updateTorchUI();
            return true;
          } catch (e) {
            console.log('applyConstraints torch failed', e);
          }
        }

        // Fallback via ImageCapture (some browsers expose fillLightMode)
        if (window.ImageCapture) {
          try {
            const ic = new ImageCapture(track);
            const pc = await ic.getPhotoCapabilities();
            if (pc && pc.fillLightMode && pc.fillLightMode.includes('torch')) {
              await track.applyConstraints({ advanced: [{ torch: on }] });
              torchOn = !!on;
              updateTorchUI();
              return true;
            }
          } catch (e) {
            console.log('ImageCapture torch failed', e);
          }
        }

        return false;
      } catch (err) {
        console.error('setTorch error', err);
        return false;
      }
    }

    async function toggleFlashlight() {
      const success = await setTorch(!torchOn);
      if (!success) alert('Flashlight not supported on this device/browser.');
    }

    function updateTorchUI() {
      const btn = document.getElementById('flashBtn');
      if (!btn) return;
      if (torchOn) {
        btn.style.background = '#ffd54f';
        btn.textContent = '🔦 On';
      } else {
        btn.style.background = '';
        btn.textContent = '🔦';
      }
    }
    
    // ============================================
    // Main Processing Loop
    // ============================================
    
    function processFrames() {
      if (!isRunning) return;
      
      // Check if video is actually playing
      if (video.paused || video.ended || !video.videoWidth) {
        requestAnimationFrame(processFrames);
        return;
      }
      
      frameCount++;
      
      // Calculate FPS
      const now = performance.now();
      if (lastFrameTime) {
        fps = 1000 / (now - lastFrameTime);
        document.getElementById('debugFps').textContent = fps.toFixed(1);
      }
      lastFrameTime = now;
      
      // Draw with high quality settings
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const roiX = Math.floor((canvas.width - ROI_WIDTH) / 2);
      const roiY = Math.floor((canvas.height - ROI_HEIGHT) / 2);
      const imageData = ctx.getImageData(roiX, roiY, ROI_WIDTH, ROI_HEIGHT);
      
      const grayData = new Uint8Array(ROI_WIDTH * ROI_HEIGHT);
      for (let i = 0; i < grayData.length; i++) {
        const r = imageData.data[i * 4];
        const g = imageData.data[i * 4 + 1];
        const b = imageData.data[i * 4 + 2];
        grayData[i] = Math.floor(0.299 * r + 0.587 * g + 0.114 * b);
      }
      
      const enhanced = applyCLAHE(grayData, ROI_WIDTH, ROI_HEIGHT);
      const power = calculateGoosebumpPower(enhanced, ROI_WIDTH, ROI_HEIGHT);
      
      document.getElementById('debugPower').textContent = power.toFixed(2);
      
      if (!baselineEstablished) {
        baselineFrames.push(power);
        
        if (baselineFrames.length >= BASELINE_FRAMES) {
          baselinePower = baselineFrames.reduce((a, b) => a + b, 0) / baselineFrames.length;
          baselineEstablished = true;
          console.log('✓ Baseline:', baselinePower);
          document.getElementById('debugBaseline').textContent = baselinePower.toFixed(2);
        }
        
        updateUI('baseline', 0, power);
      } else {
        if (baselinePower > 0) {
          currentIntensity = ((power - baselinePower) / baselinePower) * 100;
        } else {
          currentIntensity = 0;
        }
        
        const detect = currentIntensity >= DETECTION_THRESHOLD;
        if (detect) {
          detectionCount++;
          vibrate();
        }
        
        intensityHistory.push(currentIntensity);
        if (intensityHistory.length > MAX_HISTORY) {
          intensityHistory.shift();
        }
        
        updateUI(detect ? 'detecting' : 'monitoring', currentIntensity, power);
        drawGraph();
      }
      
      requestAnimationFrame(processFrames);
    }
    
    // ============================================
    // UI Updates
    // ============================================
    
    function updateUI(status, intensity, power) {
      const statusBanner = document.getElementById('statusBanner');
      const intensityDisplay = document.getElementById('intensityDisplay');
      const roiBox = document.getElementById('roiBox');
      
      statusBanner.className = 'status-banner ' + status;
      
      if (status === 'baseline') {
        statusBanner.textContent = `🔴 Baseline: ${baselineFrames.length}/${BASELINE_FRAMES}`;
        roiBox.classList.remove('detecting');
      } else if (status === 'detecting') {
        statusBanner.textContent = `🎉 GOOSEBUMPS DETECTED!`;
        roiBox.classList.add('detecting');
      } else {
        statusBanner.textContent = `👁️ Monitoring...`;
        roiBox.classList.remove('detecting');
      }
      
      intensityDisplay.textContent = intensity.toFixed(1) + '%';
      intensityDisplay.style.color = intensity > DETECTION_THRESHOLD ? '#00ff00' : '#fff';
      
      document.getElementById('statBaseline').textContent = baselinePower.toFixed(2);
      document.getElementById('statCurrent').textContent = power.toFixed(2);
      document.getElementById('statDetections').textContent = detectionCount;
    }
    
    function drawGraph() {
      const canvas = document.getElementById('graphCanvas');
      const ctx = graphCtx;
      const w = canvas.width;
      const h = canvas.height;
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.fillRect(0, 0, w, h);
      
      if (intensityHistory.length < 2) return;
      
      const thresholdY = h - (DETECTION_THRESHOLD / 100) * h;
      ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(0, thresholdY);
      ctx.lineTo(w, thresholdY);
      ctx.stroke();
      ctx.setLineDash([]);
      
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      
      const step = w / MAX_HISTORY;
      
      intensityHistory.forEach((intensity, i) => {
        const x = i * step;
        const y = h - Math.max(0, Math.min(100, intensity)) / 100 * h;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    }
    
    function vibrate() {
      if ('vibrate' in navigator) {
        navigator.vibrate([200, 100, 200]);
      }
    }
    
    function toggleGraphPanel() {
      const panel = document.getElementById('graphPanel');
      panel.classList.toggle('show');
    }
    
    function toggleDebug() {
      const debug = document.getElementById('debugInfo');
      debug.classList.toggle('show');
    }
