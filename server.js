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

//  Crop Disease Classifier
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

    fs.unlinkSync(imagePath); // Cleanup
    res.json(response.data);
  } catch (error) {
    fs.unlinkSync(imagePath);
    res.status(500).json({
      error: "Failed to classify image",
      message: error.message,
    });
  }
});

// Suggest Treatment using OpenRouter (LLaMA 3)
app.post("/treatment", async (req, res) => {
  let { diseaseName } = req.body;
  diseaseName = diseaseName.replace(/_/g, " ");

  try {
    const prompt = `You are an agricultural expert. If the disease name is 'Invalid', say "This leaf image could not be identified. Please upload a clearer image of the crop leaf." Otherwise, suggest treatment for the disease: "${diseaseName}".`;


    const aiResponse = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: "meta-llama/llama-3-8b-instruct",
        messages: [
          {
            role: "system",
            content: "You are a helpful agricultural assistant for Nigerian farmers.",
          },
          {
            role: "user",
            content: prompt,
          },
        ],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );

    const advice = aiResponse.data.choices[0].message.content;
    res.json({ treatment: advice });
  } catch (err) {
    res.status(500).json({
      error: "Treatment error",
      message: err.message,
    });
  }
});

//  Start the Server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
