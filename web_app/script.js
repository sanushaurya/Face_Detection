let VIDEO = document.getElementById("video");
let CANVAS = document.getElementById("output_canvas");
let CTX = CANVAS.getContext("2d");

let embeddings = {};
const THRESHOLD = 0.1; // Threshold for similarity
// Automatically use the current host (works for both localhost and network IP)
const API_BASE = `https://face-detection-r97v.onrender.com`;
const ATTENDANCE_LOG_INTERVAL = 30000; // 30 seconds - prevent duplicate logging
const DEFAULT_PERSON_NAME = "Person"; // Default name when no embeddings loaded

// State for tracking multiple faces
let lastVerificationTime = 0;
const VERIFY_INTERVAL = 200; // Verify every 200ms
let faceLabels = []; // Stores { box: [x,y,w,h], label: "Name", color: "green" }

// Attendance tracking
let lastAttendanceLog = {}; // { personName: timestamp }
let attendanceStartTime = {}; // { personName: startTime }

// Resize canvas to full screen
function resizeCanvas() {
  CANVAS.width = window.innerWidth;
  CANVAS.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Load embeddings from API
async function loadEmbeddings() {
  try {
    const res = await fetch(`${API_BASE}/api/embeddings`);
    if (res.ok) {
      const embeddingsList = await res.json();

      for (let item of embeddingsList) {
        try {
          const embRes = await fetch(`embeddings/${item.path.split('/')[1]}`);
          if (embRes.ok) {
            const data = await embRes.json();
            embeddings[item.name] = new Float32Array(data.embedding);
            console.log(`Loaded embedding for ${item.name}`);
          }
        } catch (e) {
          console.warn(`Could not load embedding for ${item.name}`);
        }
      }
    }
  } catch (e) {
    console.warn('Could not load embeddings from API, trying fallback...');
    // Fallback to local embeddings
    const persons = ["Aditya"];
    for (let p of persons) {
      try {
        const res = await fetch(`embeddings/${p}_embedding.json`);
        if (res.ok) {
          const data = await res.json();
          embeddings[p] = new Float32Array(data.embedding);
          console.log(`Loaded embedding for ${p}`);
        }
      } catch (e) {
        console.warn(`Could not load ${p} embedding`);
      }
    }
  }
}
loadEmbeddings();

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function extractEmbedding(landmarks) {
  let arr = [];
  for (let lm of landmarks) arr.push(lm.x, lm.y, lm.z);

  function dist(a, b) {
    return Math.hypot(
      landmarks[a].x - landmarks[b].x,
      landmarks[a].y - landmarks[b].y,
      landmarks[a].z - landmarks[b].z
    );
  }

  const KP = {
    leftEye: 33, rightEye: 263, nose: 1, chin: 152,
    leftMouth: 61, rightMouth: 291
  };

  let derived = [
    dist(KP.leftEye, KP.rightEye), dist(KP.leftEye, KP.nose),
    dist(KP.rightEye, KP.nose), dist(KP.nose, KP.chin),
    dist(KP.leftMouth, KP.rightMouth), dist(KP.leftEye, KP.chin),
    dist(KP.rightEye, KP.chin)
  ];

  let raw = new Float32Array(arr);
  let der = new Float32Array(derived);

  raw = raw.map(v => v / (Math.sqrt(raw.reduce((a, b) => a + b * b, 0)) || 1));
  der = der.map(v => v / (Math.sqrt(der.reduce((a, b) => a + b * b, 0)) || 1));

  let final = new Float32Array(raw.length + der.length);
  final.set(raw);
  final.set(der, raw.length);
  return final;
}

function identifyFace(landmarks) {
  const emb = extractEmbedding(landmarks);
  let bestPerson = DEFAULT_PERSON_NAME;
  let bestSim = -1;

  // If embeddings exist, try to match
  if (Object.keys(embeddings).length > 0) {
    for (let p in embeddings) {
      let sim = cosine(emb, embeddings[p]);
      if (sim > bestSim) {
        bestSim = sim;
        bestPerson = p;
      }
    }

    if (bestSim >= THRESHOLD) {
      return { name: bestPerson.toUpperCase(), color: "#00FF00", confidence: bestSim }; // Green - Known
    } else {
      return { name: DEFAULT_PERSON_NAME, color: "#FF0000", confidence: bestSim }; // Red - Unknown
    }
  } else {
    // No embeddings loaded, use default "Person"
    return { name: DEFAULT_PERSON_NAME, color: "#FFA500", confidence: 0.5 }; // Orange - Default
  }
}

function drawResults(results) {
  CTX.clearRect(0, 0, CANVAS.width, CANVAS.height);

  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    return;
  }

  const now = Date.now();
  const shouldVerify = (now - lastVerificationTime > VERIFY_INTERVAL);
  if (shouldVerify) {
    lastVerificationTime = now;
    faceLabels = []; // Reset labels for this frame
  }

  // Loop through all detected faces
  results.multiFaceLandmarks.forEach((landmarks, index) => {
    // Calculate bounding box
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (let lm of landmarks) {
      if (lm.x < minX) minX = lm.x;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.y > maxY) maxY = lm.y;
    }

    const x = minX * CANVAS.width;
    const y = minY * CANVAS.height;
    const w = (maxX - minX) * CANVAS.width;
    const h = (maxY - minY) * CANVAS.height;

    // Identify Face (only if we are verifying this frame, OR use cached label if available)
    // Since face index order can change, we'll just re-verify every interval.
    // For smoother UI, if we are NOT verifying, we might want to skip drawing or just draw box.
    // But to keep it simple and responsive:

    let labelData;
    if (shouldVerify) {
      labelData = identifyFace(landmarks);
      faceLabels.push(labelData); // Cache it (simple cache, assumes order stays same for 200ms)

      // Log attendance if person is identified and not recently logged
      if (labelData.name !== "Unknown" && labelData.name !== "Loading...") {
        logAttendance(labelData.name, labelData.confidence);
      }
    } else {
      // Use cached label if exists, else default
      labelData = faceLabels[index] || { name: "...", color: "gray", confidence: 0 };
    }

    // Draw Box
    CTX.strokeStyle = labelData.color;
    CTX.lineWidth = 4;
    CTX.strokeRect(x, y, w, h);

    // Draw Text Background
    const text = labelData.name;
    const startTime = attendanceStartTime[labelData.name];
    const timeText = startTime ? `${text} (${startTime.toLocaleTimeString()})` : text;

    CTX.font = "bold 24px Arial";
    const textWidth = CTX.measureText(timeText).width;

    CTX.fillStyle = labelData.color;
    CTX.fillRect(x, y - 30, textWidth + 20, 30);

    // Draw Text
    CTX.fillStyle = "black";
    CTX.fillText(timeText, x + 10, y - 7);
  });
}

async function logAttendance(personName, confidence) {
  const now = Date.now();
  const lastLog = lastAttendanceLog[personName] || 0;

  // Prevent duplicate logging within ATTENDANCE_LOG_INTERVAL
  if (now - lastLog < ATTENDANCE_LOG_INTERVAL) {
    return;
  }

  lastAttendanceLog[personName] = now;

  // Track start time for this person
  if (!attendanceStartTime[personName]) {
    attendanceStartTime[personName] = new Date();
  }

  try {
    const response = await fetch(`${API_BASE}/api/attendance`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        name: personName,
        confidence: confidence,
        start_time: attendanceStartTime[personName].toISOString()
      })
    });

    if (response.ok) {
      const data = await response.json();
      console.log(`✓ Attendance logged for ${personName} at ${new Date().toLocaleTimeString()}`);

      // Show notification
      showAttendanceNotification(personName, data);
    } else {
      console.error('Failed to log attendance:', response.statusText);
    }
  } catch (error) {
    console.error('Error logging attendance:', error);
  }
}

function showAttendanceNotification(personName, record) {
  // Create notification element
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #28a745;
    color: white;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 1000;
    font-weight: bold;
    animation: slideIn 0.3s ease-out;
  `;

  const time = new Date(record.timestamp).toLocaleTimeString();
  notification.innerHTML = `✓ ${personName} logged at ${time}`;

  document.body.appendChild(notification);

  // Remove after 3 seconds
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease-out';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Start camera + face mesh
async function startCamera() {
  let faceMesh = new FaceMesh({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    refineLandmarks: true,
    maxNumFaces: 4, // Allow multiple faces
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  faceMesh.onResults(drawResults);

  let stream = await navigator.mediaDevices.getUserMedia({ video: true });
  VIDEO.srcObject = stream;
  await VIDEO.play();

  const camera = new Camera(VIDEO, {
    onFrame: async () => {
      await faceMesh.send({ image: VIDEO });
    },
    width: 640,
    height: 480
  });

  camera.start();
}

startCamera();

// Global variable for registration usage
let latestLandmarksForSave = null;
const originalDrawResults = drawResults;
drawResults = (results) => {
  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    latestLandmarksForSave = results.multiFaceLandmarks[0];
  } else {
    latestLandmarksForSave = null;
  }
  originalDrawResults(results);
};

// SAVE BUTTON (Only on Register Page)
const saveBtn = document.getElementById("saveBtn");
if (saveBtn) {
  saveBtn.onclick = async () => {
    const nameInput = document.getElementById("personName");
    const name = nameInput.value.trim();

    if (!name) {
      alert("Please enter a name first!");
      return;
    }

    if (!latestLandmarksForSave) {
      alert("No face detected! Please look at the camera.");
      return;
    }

    saveBtn.disabled = true;
    saveBtn.textContent = "Saving...";

    try {
      const emb = extractEmbedding(latestLandmarksForSave);
      const embArray = Array.from(emb);

      const response = await fetch(`${API_BASE}/api/embeddings/${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          embedding: embArray
        })
      });

      if (response.ok) {
        alert(`✓ Embedding saved for ${name}!\n\nThe system will now recognize this person.`);
        nameInput.value = '';
        // Reload embeddings
        loadEmbeddings();
      } else {
        const error = await response.json();
        alert(`Error: ${error.error || 'Failed to save embedding'}`);
      }
    } catch (error) {
      console.error('Error saving embedding:', error);
      alert(`Error saving embedding: ${error.message}`);
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = "Save Embedding";
    }
  };
}
