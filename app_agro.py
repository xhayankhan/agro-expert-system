# app.py
from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image
import whisper
import base64
import io
import os
import time

app = Flask(__name__)

# Create folders
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/audio", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading models...")

# Load models
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

model = PeftModel.from_pretrained(base_model, "models/agro-expert").to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

whisper_model = whisper.load_model("base")

print("AgroExpert Vision READY!")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AgroExpert Vision</title>
    <style>
        :root {
            --primary: #28a745;
            --dark: #1e3d1e;
        }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0; padding: 20px; min-height: 100vh;
        }
        .container {
            max-width: 1200px; margin: 0 auto;
            background: white; border-radius: 20px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, var(--primary), var(--dark));
            color: white; padding: 30px; text-align: center;
        }
        header h1 { margin: 0; font-size: 2.8em; }
        .content { padding: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

        .box {
            background: #f8fff9; padding: 25px; border-radius: 15px;
            border: 2px solid #e0f2e6; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        textarea {
            width: 100%; padding: 15px; border: 2px solid #ddd;
            border-radius: 10px; font-size: 16px; resize: vertical;
            min-height: 180px;
        }
        textarea:focus { outline: none; border-color: var(--primary); }

        .upload-area {
            border: 3px dashed var(--primary); border-radius: 15px;
            padding: 40px; text-align: center; cursor: pointer;
            background: #f0f9f0; transition: 0.3s;
        }
        .upload-area:hover { background: #e8f5e9; }
        .preview { margin-top: 15px; max-height: 300px; border-radius: 10px; }

        button {
            background: var(--primary); color: white; border: none;
            padding: 15px 30px; margin: 10px 5px; border-radius: 10px;
            font-size: 16px; cursor: pointer; font-weight: bold;
        }
        button:hover { background: #218838; transform: translateY(-2px); }
        .example-btn {
            background: #e3f2fd; color: #1976d2; border: 1px solid #bbdefb;
            font-size: 14px; padding: 10px 15px;
        }
        .example-btn:hover { background: #bbdefb; }

        .result {
            margin-top: 30px; padding: 25px; background: #f8fff8;
            border-radius: 15px; border-left: 6px solid var(--primary);
            white-space: pre-wrap; line-height: 1.8;
        }
        .loading { text-align: center; padding: 40px; color: var(--primary); }
        .spinner {
            border: 5px solid #f3f3f3; border-top: 5px solid var(--primary);
            border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>AgroExpert Vision</h1>
        <p>AI Assistant for Farmers • Image + Voice + Text</p>
    </header>

    <div class="content">
        <div class="grid">
            <div class="box">
                <h3>Upload Plant Image</h3>
                <div class="upload-area" onclick="document.getElementById('img').click()"
                     ondrop="drop(event)" ondragover="allowDrop(event)">
                    <p>Drop image here or click to upload</p>
                    <input type="file" id="img" accept="image/*" onchange="preview(this)" style="display:none">
                </div>
                <div id="preview"></div>
            </div>

            <div class="box">
                <h3>Describe Problem or Record Voice</h3>
                <textarea id="text" placeholder="e.g. Tomato leaves have yellow spots and curling..."></textarea>
                <br><br>
                <button onclick="startRecord()">Record Voice</button>
                <button onclick="stopRecord()" id="stopBtn" style="display:none; background:#dc3545">Stop</button>
                <span id="status"></span>

                <div style="margin-top:20px">
                    <strong>Quick Examples:</strong><br>
                    <button class="example-btn" onclick="setText('My tomato leaves have yellow spots and curling')">Tomato Disease</button>
                    <button class="example-btn" onclick="setText('White insects under cotton leaves')">Cotton Pest</button>
                    <button class="example-btn" onclick="setText('What fertilizer for wheat in sandy soil?')">Wheat Fertilizer</button>
                </div>
            </div>
        </div>

        <div style="text-align:center; margin-top:30px">
            <button onclick="analyze()" style="font-size:20px; padding:18px 50px">Analyze & Get Advice</button>
            <button onclick="clearAll()">Clear All</button>
        </div>

        <div id="result"></div>
    </div>
</div>

<script>
let recorder, audioBlob;
let imageFile = null;

function preview(input) {
    if (input.files && input.files[0]) {
        imageFile = input.files[0];
        const reader = new FileReader();
        reader.onload = e => document.getElementById('preview').innerHTML = 
            `<img src="${e.target.result}" class="preview">`;
        reader.readAsDataURL(imageFile);
    }
}

function allowDrop(e) { e.preventDefault(); }
function drop(e) {
    e.preventDefault();
    imageFile = e.dataTransfer.files[0];
    const reader = new FileReader();
    reader.onload = ev => document.getElementById('preview').innerHTML = 
        `<img src="${ev.target.result}" class="preview">`;
    reader.readAsDataURL(imageFile);
}

async function startRecord() {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
        audioBlob = new Blob(chunks, {type:'audio/wav'});
        transcribe();
    };
    recorder.start();
    document.getElementById('stopBtn').style.display = 'inline-block';
    document.getElementById('status').innerText = 'Recording...';
}

function stopRecord() {
    recorder.stop();
    recorder.stream.getTracks().forEach(t => t.stop());
    document.getElementById('stopBtn').style.display = 'none';
    document.getElementById('status').innerText = 'Transcribing...';
}

async function transcribe() {
    const reader = new FileReader();
    reader.onload = async () => {
        const resp = await fetch('/transcribe', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({audio: reader.result.split(',')[1]})
        });
        const data = await resp.json();
        document.getElementById('text').value = data.text;
        document.getElementById('status').innerText = 'Transcribed!';
    };
    reader.readAsDataURL(audioBlob);
}

function setText(txt) {
    document.getElementById('text').value = txt;
}

async function analyze() {
    const text = document.getElementById('text').value;
    let image = null;
    if (imageFile) {
        const reader = new FileReader();
        image = await new Promise(resolve => {
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(imageFile);
        });
    }

    document.getElementById('result').innerHTML = `
        <div class="loading"><div class="spinner"></div><p>Analyzing... Please wait 10-20 seconds</p></div>`;

    const resp = await fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text, image})
    });
    const data = await resp.json();

    let html = '';
    if (data.image_analysis) {
        html += `<div style="background:#fff3cd;padding:15px;border-radius:10px;margin-bottom:15px">
                    <strong>Image Analysis:</strong><br>${data.image_analysis}</div>`;
    }
    html += `<div class="result"><strong>Expert Advice:</strong><br><br>${data.advice}</div>`;
    document.getElementById('result').innerHTML = html;
}

function clearAll() {
    document.getElementById('text').value = '';
    document.getElementById('preview').innerHTML = '';
    document.getElementById('result').innerHTML = '';
    imageFile = null;
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    image_b64 = data.get('image')

    context = text

    if image_b64:
        try:
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            inputs = blip_processor(img, "a photo of a plant with", return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_new_tokens=60)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)

            context = f"Image shows: {caption}. Farmer says: {text}" if text else caption
        except:
            caption = "Image uploaded"

    prompt = f"""<|system|>
You are an expert agricultural advisor for Indian farmers. Give practical, safe advice in simple language.</s>
<|user|>
{context}</s>
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True, repetition_penalty=1.2)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response.split("<|assistant|>")[-1].strip()

    return jsonify({
        "image_analysis": caption if image_b64 else None,
        "advice": answer
    })


@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_b64 = request.json['audio']
    audio_bytes = base64.b64decode(audio_b64)

    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    result = whisper_model.transcribe("temp.wav")
    os.remove("temp.wav")

    return jsonify({"text": result["text"]})


# OLD (default port 5000)
# app.run(host='127.0.0.1', port=5000, debug=False)

# NEW → Use port 7860
if __name__ == '__main__':
    print("Open → http://127.0.0.1:7860")
    app.run(host='0.0.0.0', port=7860, debug=False)