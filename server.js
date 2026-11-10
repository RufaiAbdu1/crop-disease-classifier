require("dotenv").config();
const express = require("express");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const app = express();
const PORT = process.env.PORT || 3000;

// ---------- Setup ----------
const UPLOAD_DIR = path.join(__dirname, "uploads");
const DATA_DIR = path.join(__dirname, "data");
fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(DATA_DIR, { recursive: true });

const CLASSIFY_LOG = path.join(DATA_DIR, "classifications.jsonl");
const FEEDBACK_LOG = path.join(DATA_DIR, "feedback.jsonl");

const HF_MODEL_URL =
  process.env.HF_MODEL_URL ||
  "https://api-inference.huggingface.co/models/wambugu71/crop_leaf_diseases_vit";

const DEFAULT_SPACE = "RufaiAbdu/easifarm-classifier";
const rawSpace = String(process.env.HF_SPACE || "").trim().replace(/^["']|["']$/g, "").replace(/\/+$/, "");
const HF_SPACE = /^[\w.-]+\/[\w.-]+$/.test(rawSpace) || /^https:\/\/[^\s]+$/.test(rawSpace)
  ? rawSpace
  : DEFAULT_SPACE;
console.log("Using Space:", HF_SPACE, "| env value was:", JSON.stringify(process.env.HF_SPACE));
console.log(
  "HF token present:", Boolean(process.env.HUGGINGFACE_API_KEY),
  "| starts with hf_:", String(process.env.HUGGINGFACE_API_KEY || "").startsWith("hf_")
);

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 8 * 1024 * 1024 }, // 8 MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype && file.mimetype.startsWith("image/")) return cb(null, true);
    cb(new Error("Only image files are allowed."));
  },
});

app.use(express.static("public"));
app.use(express.json());

// ---------- Helpers ----------
function safeUnlink(p) {
  try {
    if (p && fs.existsSync(p)) fs.unlinkSync(p);
  } catch (e) {
    console.error("Cleanup failed:", e.message);
  }
}

function appendLog(file, obj) {
  try {
    fs.appendFileSync(file, JSON.stringify(obj) + "\n");
  } catch (e) {
    console.error("Log write failed:", e.message);
  }
}

function readLog(file) {
  try {
    return fs
      .readFileSync(file, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch (e) {
    return [];
  }
}

function requireAdmin(req, res, next) {
  const key = process.env.ADMIN_KEY;
  if (!key) return res.status(403).send("Set ADMIN_KEY in your environment to enable this.");
  if (req.query.key !== key) return res.status(401).send("Wrong key.");
  next();
}

// Connects to a Hugging Face Space (Gradio) when HF_SPACE is set, e.g. "yourname/easifarm-classifier"
let gradioClientPromise = null;
function getGradioClient() {
  if (!gradioClientPromise) {
    gradioClientPromise = import("@gradio/client")
      .then(({ Client }) =>
        Client.connect(HF_SPACE, { hf_token: process.env.HUGGINGFACE_API_KEY })
      )
      .catch((e) => {
        gradioClientPromise = null; // allow a retry on the next request
        throw e;
      });
  }
  return gradioClientPromise;
}

// ---------- AI vision fallback (OpenRouter) ----------
// Used automatically if the Hugging Face Space fails. Change the model in Render with VISION_MODEL.
const VISION_MODEL = process.env.VISION_MODEL || "google/gemini-3.5-flash";
const TREATMENT_MODEL = process.env.TREATMENT_MODEL || VISION_MODEL;

async function classifyWithVision(imageBuffer, mimetype) {
  const dataUrl = `data:${mimetype || "image/jpeg"};base64,${imageBuffer.toString("base64")}`;
  const prompt =
    "You help smallholder farmers in Nigeria. Look at this crop leaf photo and reply ONLY with JSON, no other text: " +
    '{"crop": "crop name or Unknown", "condition": "disease name, or Healthy, or Invalid if this is not a plant leaf", "confidence": a number from 0 to 1}. ' +
    "Be conservative: if the photo is blurry, far away, or you are unsure, use a low confidence.";

  const resp = await axios.post(
    "https://openrouter.ai/api/v1/chat/completions",
    {
      model: VISION_MODEL,
      temperature: 0,
      max_tokens: 200,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            { type: "image_url", image_url: { url: dataUrl } },
          ],
        },
      ],
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      timeout: 60000,
    }
  );

  const text = resp.data.choices[0].message.content || "";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("Vision model returned no JSON: " + text.slice(0, 100));
  const out = JSON.parse(match[0]);

  const condition = String(out.condition || "Invalid").trim();
  const crop = String(out.crop || "").trim();
  let score = Number(out.confidence);
  if (!isFinite(score)) score = 0.3;
  score = Math.max(0, Math.min(1, score));

  const label = /^invalid$/i.test(condition)
    ? "Invalid"
    : crop && !/^unknown$/i.test(crop)
    ? `${crop} - ${condition}`
    : condition;
  return [{ label, score }];
}

// ---------- 1. Classify the leaf image ----------
app.post("/classify", (req, res) => {
  upload.single("image")(req, res, async (uploadErr) => {
    if (uploadErr) {
      return res.status(400).json({ error: "Upload error", message: uploadErr.message });
    }
    if (!req.file) {
      return res
        .status(400)
        .json({ error: "No image", message: "Please choose a leaf photo first." });
    }

    const imagePath = req.file.path;
    const startedAt = Date.now();
    console.log("Classify request received:", req.file.mimetype, req.file.size, "bytes");
    try {
      const image = fs.readFileSync(imagePath);

      let predictions = [];
      let source = "model";
      try {
        const client = await getGradioClient();
        const blob = new Blob([image], { type: req.file.mimetype || "image/jpeg" });
        console.log("Connected to Space, sending image...");
        const result = await Promise.race([
          client.predict("/predict", { image: blob }),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Space did not answer within 30 seconds")), 30000)
          ),
        ]);
        predictions = Array.isArray(result.data && result.data[0]) ? result.data[0] : [];
        if (!predictions.length) throw new Error("Space returned no predictions");
        console.log("Space answered in", ((Date.now() - startedAt) / 1000).toFixed(1), "s");
      } catch (spaceErr) {
        console.error("Space failed, using vision fallback:", spaceErr && spaceErr.message ? spaceErr.message : spaceErr);
        predictions = await classifyWithVision(image, req.file.mimetype);
        source = "vision";
        console.log("Vision fallback answered in", ((Date.now() - startedAt) / 1000).toFixed(1), "s");
      }
      const top = predictions[0];
      const id = crypto.randomUUID();

      if (top) {
        appendLog(CLASSIFY_LOG, {
          id,
          time: new Date().toISOString(),
          label: top.label,
          confidence: top.score,
          source,
        });
      }

      res.json({ id, predictions, source });
    } catch (error) {
      const status = error.response && error.response.status;
      if (status === 503) {
        return res.status(503).json({
          error: "Model loading",
          message: "The AI model is waking up. Please try again in 20 seconds.",
        });
      }
      console.error("Classify error:", error && error.stack ? error.stack : error);
      res.status(500).json({
        error: "Failed to classify image",
        message: "Could not classify this image right now. Please try again.",
      });
    } finally {
      safeUnlink(imagePath);
    }
  });
});

// ---------- 2. Treatment advice (English or Hausa) ----------
app.post("/treatment", async (req, res) => {
  let { diseaseName, lang } = req.body || {};
  if (!diseaseName || typeof diseaseName !== "string") {
    return res.status(400).json({ error: "Missing disease name" });
  }
  diseaseName = diseaseName.replace(/_/g, " ").slice(0, 100);
  const replyLanguage = lang === "ha" ? "simple Hausa" : "simple English";

  try {
    const prompt =
      `A farmer in Nigeria uploaded a crop leaf photo. The AI diagnosis is: "${diseaseName}".\n` +
      `If the diagnosis is 'Invalid', say the image could not be identified and ask for a clearer leaf photo.\n` +
      `If the leaf looks healthy, give 3 short tips to keep the crop healthy.\n` +
      `Otherwise give practical advice in at most 6 short bullet points: what it is, what to do now, ` +
      `low-cost or cultural control first, and how to prevent it next season. ` +
      `If you mention any chemical, tell the farmer to check with an agricultural extension officer or agro-dealer first. ` +
      `Do not use markdown symbols like ** or #. Reply in ${replyLanguage}.`;

    const aiResponse = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: TREATMENT_MODEL,
        temperature: 0.3,
        max_tokens: 450,
        messages: [
          {
            role: "system",
            content: "You are a careful, helpful agricultural assistant for smallholder farmers in Nigeria.",
          },
          { role: "user", content: prompt },
        ],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
        },
        timeout: 60000,
      }
    );

    const advice = aiResponse.data.choices[0].message.content;
    res.json({ treatment: advice });
  } catch (err) {
    console.error("Treatment error:", err.message);
    res.status(500).json({
      error: "Treatment error",
      message: "Could not get treatment advice right now. Please try again or contact an expert.",
    });
  }
});

// ---------- 3. Farmer feedback: "Was this correct?" ----------
app.post("/feedback", (req, res) => {
  const { id, correct } = req.body || {};
  if (!id || typeof correct !== "boolean") {
    return res.status(400).json({ error: "id and correct (true/false) are required" });
  }
  appendLog(FEEDBACK_LOG, {
    id: String(id).slice(0, 64),
    time: new Date().toISOString(),
    correct,
  });
  res.json({ ok: true });
});

// ---------- 4. Pilot evidence (protected by ADMIN_KEY) ----------
app.get("/stats", requireAdmin, (req, res) => {
  const classifications = readLog(CLASSIFY_LOG);
  const feedback = readLog(FEEDBACK_LOG);

  const latestFeedback = new Map();
  feedback.forEach((f) => latestFeedback.set(f.id, f.correct));

  const answered = latestFeedback.size;
  const correctCount = [...latestFeedback.values()].filter(Boolean).length;

  const byLabel = {};
  classifications.forEach((c) => {
    byLabel[c.label] = (byLabel[c.label] || 0) + 1;
  });

  res.json({
    totalChecks: classifications.length,
    feedbackReceived: answered,
    markedCorrect: correctCount,
    markedWrong: answered - correctCount,
    percentCorrectOfAnswered: answered ? Math.round((correctCount / answered) * 100) : null,
    checksByDiagnosis: byLabel,
  });
});

app.get("/export.csv", requireAdmin, (req, res) => {
  const classifications = readLog(CLASSIFY_LOG);
  const latestFeedback = new Map();
  readLog(FEEDBACK_LOG).forEach((f) => latestFeedback.set(f.id, f.correct));

  const rows = ["id,time,diagnosis,confidence,source,farmer_said_correct"];
  classifications.forEach((c) => {
    const fb = latestFeedback.has(c.id) ? (latestFeedback.get(c.id) ? "yes" : "no") : "";
    const label = String(c.label).replace(/"/g, '""');
    rows.push(`${c.id},${c.time},"${label}",${(c.confidence * 100).toFixed(1)},${c.source || "model"},${fb}`);
  });

  res.setHeader("Content-Type", "text/csv");
  res.setHeader("Content-Disposition", "attachment; filename=easifarm-pilot.csv");
  res.send(rows.join("\n"));
});

// ---------- Start ----------
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
