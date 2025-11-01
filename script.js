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
    let lastFrameTime = 0;
    let fps = 0;
    let torchOn = false;
    let currentStream = null;
    let currentFacingMode = 'environment';
    
    const BASELINE_FRAMES = 10;
    const DETECTION_THRESHOLD = 30.0;
    const ROI_WIDTH = 160;
    const ROI_HEIGHT = 120;
    const MAX_HISTORY = 60;
    const CLAHE_CLIP = 2.0;
    const FREQ_MIN_MM = 0.23;
    const FREQ_MAX_MM = 0.75;
    const PIXEL_SIZE_MM = 0.25;
    
    // Error handling
    function showAppError(msg) {
      const panel = document.getElementById('errorPanel');
      const text = document.getElementById('errorText');
      text.textContent = msg;
      panel.style.display = 'block';
    }

    window.addEventListener('error', e => {
      try { showAppError(e.message + '\n' + (e.filename || '') + ':' + (e.lineno || '')); } catch (err) { console.error(err); }
    });

    window.addEventListener('unhandledrejection', e => {
      try { showAppError((e.reason?.message || e.reason) || 'Unhandled rejection'); } catch (err) { console.error(err); }
    });
    
    // FFT Implementation
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
    
    // CHILLER Algorithm
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
    
    // CLAHE Enhancement
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
    
    // Initialize
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
      graphCanvas.height = 140;
    });
    
    // Camera Setup with Flashlight Support
    async function pickRearCameraDeviceId() {
      try { await navigator.mediaDevices.getUserMedia({ video: true }); } catch {}
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter(d => d.kind === 'videoinput');
      const rear = videoInputs.find(d => /back|rear|environment/i.test(d.label || ''));
      return (rear || videoInputs[videoInputs.length - 1] || {}).deviceId;
    }

    async function startCamera() {
      const statusBanner = document.getElementById('statusBanner');
      const statusText = document.getElementById('statusText');
      try {
        statusText.textContent = 'Requesting camera...';

        if (!window.isSecureContext) {
          throw new Error('Page must be loaded over HTTPS');
        }
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Camera API not supported');
        }

        video.setAttribute('playsinline', '');
        video.setAttribute('autoplay', '');
        video.muted = true;

        let deviceId;
        try {
            deviceId = await pickRearCameraDeviceId();
        } catch (e) {
            console.log('Failed to get specific device ID, falling back to default camera');
        }

        // Try different camera configurations in order of preference
        const constraints = [
            // First try with specific resolution ladders
            {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    facingMode: deviceId ? undefined : currentFacingMode,
                    frameRate: { ideal: 30 }
                }
            },
            // Then try with just facing mode
            {
                video: {
                    facingMode: currentFacingMode
                }
            },
            // Finally try with just basic video
            {
                video: true
            }
        ];

        let stream;
        let lastError;
        // Try each constraint configuration
        for (const constraint of constraints) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraint);
                if (stream) break;
            } catch (e) {
                lastError = e;
                console.log('Failed to get stream with constraint:', constraint, e);
            }
        }

        if (!stream) {
            throw new Error('Could not start video source: ' + (lastError?.message || 'Unknown error'));
        }

        // Set up video element with the stream
        video = document.querySelector('video');
        if (!video) {
            throw new Error('Video element not found');
        }
        
        currentStream = stream;
        video.srcObject = stream;
        
        // Wait for video to be ready
        try {
            await video.play();
        } catch (e) {
            throw new Error('Failed to play video: ' + e.message);
        }

        // Apply advanced constraints
        const track = stream.getVideoTracks()[0];
        const caps = track.getCapabilities?.() || {};
        const advanced = [];

        if (caps.focusMode && caps.focusMode.includes('continuous')) {
          advanced.push({ focusMode: 'continuous' });
        }
        if (caps.exposureMode && caps.exposureMode.includes('continuous')) {
          advanced.push({ exposureMode: 'continuous' });
        }
        if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes('continuous')) {
          advanced.push({ whiteBalanceMode: 'continuous' });
        }
        if (caps.frameRate) {
          advanced.push({ frameRate: Math.min(60, caps.frameRate.max || 30) });
        }
        if (caps.width && caps.height) {
          advanced.push({
            width: Math.min(3840, caps.width.max || 1920),
            height: Math.min(2160, caps.height.max || 1080),
          });
        }

        if (advanced.length) {
          try { await track.applyConstraints({ advanced }); } catch (e) {}
        }

        isRunning = true;
        document.getElementById('startBtn').style.display = 'none';
        document.getElementById('stopBtn').style.display = 'block';
        document.getElementById('instructions').style.display = 'none';

        await new Promise(r => {
          if (video.readyState >= 1) return r();
          video.onloadedmetadata = () => r();
          setTimeout(r, 1000);
        });

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const settings = track.getSettings();
        document.getElementById('debugRes').textContent = `${video.videoWidth}×${video.videoHeight}`;
        statusText.textContent = `Camera: ${video.videoWidth}×${video.videoHeight}`;

        processFrames();

        // Check torch capability
        checkTorchSupport();

      } catch (error) {
        console.error('Camera setup error:', error);
        const errorMsg = (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError')
          ? 'Camera access denied. Enable camera in settings and refresh.'
          : `Camera error: ${error.message}`;
        alert(errorMsg);
        statusText.textContent = 'Camera error';
        statusBanner.classList.add('baseline');
      }
    }

    async function checkTorchSupport() {
      try {
        if (!currentStream) return;
        const track = currentStream.getVideoTracks()[0];
        const caps = track.getCapabilities?.() || {};
        
        let hasTorch = false;
        
        if ('torch' in caps) {
          hasTorch = true;
          document.getElementById('debugTorch').textContent = 'Available';
        } else if (window.ImageCapture) {
          try {
            const ic = new ImageCapture(track);
            const pc = await ic.getPhotoCapabilities();
            if (pc?.fillLightMode?.includes('torch')) {
              hasTorch = true;
              document.getElementById('debugTorch').textContent = 'Available (IC)';
            }
          } catch (e) {}
        }
        
        if (hasTorch) {
          document.getElementById('flashBtn').style.display = 'flex';
        } else {
          document.getElementById('debugTorch').textContent = 'Not available';
        }
      } catch (e) {
        console.log('Torch detection failed', e);
        document.getElementById('debugTorch').textContent = 'Error';
      }
    }

    async function setTorch(on) {
      try {
        if (!currentStream) return false;
        const track = currentStream.getVideoTracks()[0];
        if (!track) return false;

        const caps = track.getCapabilities?.() || {};
        
        // Method 1: Direct torch constraint
        if ('torch' in caps) {
          try {
            await track.applyConstraints({ 
              advanced: [{ torch: on }] 
            });
            torchOn = !!on;
            updateTorchUI();
            document.getElementById('debugTorch').textContent = on ? 'ON' : 'OFF';
            return true;
          } catch (e) {
            console.log('Method 1 failed:', e);
          }
        }

        // Method 2: ImageCapture API
        if (window.ImageCapture) {
          try {
            const ic = new ImageCapture(track);
            const pc = await ic.getPhotoCapabilities();
            if (pc?.fillLightMode?.includes('torch')) {
              await track.applyConstraints({
                advanced: [{ fillLightMode: on ? 'torch' : 'off' }]
              });
              torchOn = !!on;
              updateTorchUI();
              document.getElementById('debugTorch').textContent = on ? 'ON (IC)' : 'OFF';
              return true;
            }
          } catch (e) {
            console.log('Method 2 failed:', e);
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
      if (!success) {
        alert('Flashlight not supported on this device/browser. Try Chrome or Safari on a mobile device with rear camera.');
      }
    }

    function updateTorchUI() {
      const btn = document.getElementById('flashBtn');
      if (!btn) return;
      if (torchOn) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    }

    async function switchCamera() {
      if (!isRunning) return;
      
      stopCamera();
      currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
      await new Promise(r => setTimeout(r, 300));
      startCamera();
    }

    function stopCamera() {
      isRunning = false;
      if (torchOn) {
        setTorch(false);
      }
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
      }
      document.getElementById('startBtn').style.display = 'block';
      document.getElementById('stopBtn').style.display = 'none';
      document.getElementById('flashBtn').style.display = 'none';
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
    
    // Main Processing Loop
    function processFrames() {
      if (!isRunning) return;
      
      if (video.paused || video.ended || !video.videoWidth) {
        requestAnimationFrame(processFrames);
        return;
      }
      
      frameCount++;
      
      const now = performance.now();
      if (lastFrameTime) {
        fps = 1000 / (now - lastFrameTime);
        document.getElementById('debugFps').textContent = fps.toFixed(1);
      }
      lastFrameTime = now;
      
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
    
    // UI Updates
    function updateUI(status, intensity, power) {
      const statusBanner = document.getElementById('statusBanner');
      const statusText = document.getElementById('statusText');
      const intensityDisplay = document.getElementById('intensityDisplay');
      const roiBox = document.getElementById('roiBox');
      
      statusBanner.className = 'status-banner ' + status;
      
      if (status === 'baseline') {
        statusText.textContent = `Calibrating: ${baselineFrames.length}/${BASELINE_FRAMES}`;
        roiBox.classList.remove('detecting');
        intensityDisplay.classList.remove('high');
      } else if (status === 'detecting') {
        statusText.textContent = `🎉 GOOSEBUMPS DETECTED!`;
        roiBox.classList.add('detecting');
        intensityDisplay.classList.add('high');
      } else {
        statusText.textContent = `Monitoring...`;
        roiBox.classList.remove('detecting');
        intensityDisplay.classList.remove('high');
      }
      
      // Use default values when values are undefined or reset
      intensityDisplay.textContent = (intensity || 0).toFixed(1) + '%';
      
      document.getElementById('statBaseline').textContent = (baselinePower || 0).toFixed(2);
      document.getElementById('statCurrent').textContent = (power || 0).toFixed(2);
      document.getElementById('statDetections').textContent = detectionCount || 0;
    }
    
    function drawGraph() {
      const canvas = document.getElementById('graphCanvas');
      const ctx = graphCtx;
      const w = canvas.width;
      const h = canvas.height;
      
      // Clear with gradient background
      const gradient = ctx.createLinearGradient(0, 0, 0, h);
      gradient.addColorStop(0, 'rgba(15, 15, 30, 0.95)');
      gradient.addColorStop(1, 'rgba(26, 26, 46, 0.95)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, w, h);
      
      if (intensityHistory.length < 2) return;
      
      // Draw grid lines
      ctx.strokeStyle = 'rgba(79, 209, 197, 0.1)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = (h / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
      
      // Draw threshold line
      const thresholdY = h - (DETECTION_THRESHOLD / 100) * h;
      ctx.strokeStyle = 'rgba(252, 129, 129, 0.6)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 4]);
      ctx.beginPath();
      ctx.moveTo(0, thresholdY);
      ctx.lineTo(w, thresholdY);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Draw area under curve
      ctx.fillStyle = 'rgba(79, 209, 197, 0.15)';
      ctx.beginPath();
      const step = w / MAX_HISTORY;
      ctx.moveTo(0, h);
      
      intensityHistory.forEach((intensity, i) => {
        const x = i * step;
        const y = h - Math.max(0, Math.min(100, intensity)) / 100 * h;
        ctx.lineTo(x, y);
      });
      
      ctx.lineTo((intensityHistory.length - 1) * step, h);
      ctx.closePath();
      ctx.fill();
      
      // Draw line
      const lineGradient = ctx.createLinearGradient(0, 0, w, 0);
      lineGradient.addColorStop(0, '#3182ce');
      lineGradient.addColorStop(0.5, '#4fd1c5');
      lineGradient.addColorStop(1, '#48bb78');
      ctx.strokeStyle = lineGradient;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      
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
      
      // Draw current point
      if (intensityHistory.length > 0) {
        const lastIntensity = intensityHistory[intensityHistory.length - 1];
        const lastX = (intensityHistory.length - 1) * step;
        const lastY = h - Math.max(0, Math.min(100, lastIntensity)) / 100 * h;
        
        ctx.fillStyle = lastIntensity >= DETECTION_THRESHOLD ? '#48bb78' : '#4fd1c5';
        ctx.beginPath();
        ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
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