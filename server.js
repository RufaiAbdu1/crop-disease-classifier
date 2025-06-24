require("dotenv").config();
const express = require("express");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = process.env.PORT || 3000;

// Serve frontend from public folder
app.use(express.static("public"));

// API endpoint to handle image upload and classification
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
          "Content-Type": "application/octet-stream"
        }
      }
    );

    fs.unlinkSync(imagePath); // Clean up

    res.json(response.data); // Send prediction to frontend
  } catch (error) {
    fs.unlinkSync(imagePath);
    res.status(500).json({
      error: "Failed to classify image",
      message: error.message
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
