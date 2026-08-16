import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import { InferenceClient } from "@huggingface/inference";


dotenv.config();

const app = express();
const port = process.env.PORT || 5000;
app.use(cors());
app.use(express.json());


const client = new InferenceClient(process.env.HF_ACCESS_TOKEN);



app.post("/api/sentiment", async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({
        success: false,
        message: "Text Input is Empty",
      });
    }

    let result = await client.textClassification({
      model: "cardiffnlp/twitter-roberta-base-sentiment-latest",
      inputs: text,
      provider: "auto",
    }); 

    return res.status(200).json({
      result,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

app.listen(port, () => {
  console.log(`Server running successfully`);
});
