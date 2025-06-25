require("dotenv").config();
const express = require("express");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = process.env.PORT || 3000;

app.use(express.static("public"));
app.use(express.json());

// Route: POST /classify - Send image to Hugging Face
app.post("/classify", upload.single("image"), async (req, res) => {
  const imagePath = req.file.path;

  try {
    const image = fs.readFileSync(imagePath);

    const response = await axios.post(
      "https://api-inference.huggingface.co/models/wambugu71/crop_leaf_diseases_vit",
      image,
      {
        headers: {
          Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
          "Content-Type": "application/octet-stream",
        },
      }
    );

    fs.unlinkSync(imagePath); // Clean up uploaded image

    res.json(response.data);
  } catch (error) {
    fs.unlinkSync(imagePath);
    res.status(500).json({
      error: "Failed to classify image",
      message: error.message,
    });
  }
});

// Route: POST /treatment - Use OpenRouter to generate treatment advice
app.post("/treatment", async (req, res) => {
  const { diseaseName } = req.body;

  try {
    const prompt = `Suggest practical, short, and beginner-friendly treatment advice for this crop disease: "${diseaseName}". Focus on tips suitable for rural Nigerian farmers.`;

    const response = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: "openai/gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );

    const treatment = response.data.choices[0].message.content;
    res.json({ treatment });
  } catch (err) {
    res.status(500).json({
      error: "AI treatment generation failed",
      message: err.message,
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
