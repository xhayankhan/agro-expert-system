from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image
import whisper
import os
import base64
import time
import io

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("🌾 Loading AgroExpert Vision System...\n")

# Load text model
print("1️⃣ Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

print("2️⃣ Loading AgroExpert LoRA...")
model = PeftModel.from_pretrained(base_model, "models/agro-expert").to(device)
model.eval()

print("3️⃣ Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

print("4️⃣ Loading Vision Model (BLIP)...")
# Using BLIP for image captioning/analysis
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

print("5️⃣ Loading Whisper...")
whisper_model = whisper.load_model("base").to(device)

print("\n✅ AgroExpert Vision ready!\n")

# Create directories
AUDIO_DIR = "agro_recordings"
IMAGE_DIR = "agro_images"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgroExpert Vision - Agricultural Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #52c234 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #52c234 0%, #061700 100%);
            padding: 30px;
            color: white;
            text-align: center;
        }
        .header h1 { 
            font-size: 2.5em; 
            margin-bottom: 10px;
        }
        .disclaimer {
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 15px;
            margin: 20px;
            border-radius: 5px;
        }
        .disclaimer h3 { color: #155724; margin-bottom: 10px; }
        .disclaimer ul { margin-left: 20px; color: #155724; line-height: 1.6; }
        .content { padding: 30px; }

        .input-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        @media (max-width: 768px) {
            .input-grid {
                grid-template-columns: 1fr;
            }
        }

        .input-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
        }
        .input-section h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #52c234;
        }

        .image-upload-area {
            border: 3px dashed #52c234;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            background: #f0f9f0;
            cursor: pointer;
            transition: all 0.3s;
        }
        .image-upload-area:hover {
            background: #e8f5e9;
            border-color: #28a745;
        }
        .image-upload-area.dragover {
            background: #c8e6c9;
            border-color: #28a745;
        }
        #imageInput {
            display: none;
        }
        .upload-icon {
            font-size: 48px;
            color: #52c234;
            margin-bottom: 10px;
        }
        .image-preview {
            margin-top: 15px;
            max-width: 100%;
            border-radius: 10px;
            display: none;
        }
        .image-preview img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn-primary {
            background: linear-gradient(135deg, #52c234 0%, #28a745 100%);
            color: white;
            flex: 1;
        }
        .btn-primary:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(40, 167, 69, 0.3); 
        }
        .btn-image {
            background: #ff6b35;
            color: white;
        }
        .btn-record {
            background: #17a2b8;
            color: white;
        }
        .btn-record.recording {
            background: #dc3545;
            animation: pulse 1.5s infinite;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .result-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            border: 2px solid #e9ecef;
            min-height: 300px;
        }
        .result-section h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #52c234;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #52c234;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .advice {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            white-space: pre-wrap;
            line-height: 1.8;
            font-size: 15px;
        }
        .image-analysis {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #ffc107;
        }
        .image-analysis h3 {
            color: #856404;
            margin-bottom: 10px;
        }
        .examples {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }
        .example-btn {
            display: inline-block;
            padding: 10px 15px;
            margin: 5px;
            background: #e8f5e9;
            border: 1px solid #4caf50;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        .example-btn:hover {
            background: #4caf50;
            color: white;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .feature-item {
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #52c234;
        }
        .feature-item h4 {
            color: #28a745;
            margin-bottom: 8px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌾 AgroExpert Vision System 📸</h1>
            <p>AI-Powered Agricultural Assistant with Image Analysis</p>
        </div>

        <div class="disclaimer">
            <h3>📢 How to Use This Enhanced System</h3>
            <ul>
                <li><strong>📸 Upload Images:</strong> Take photos of diseased plants, pests, or crop problems</li>
                <li><strong>💬 Describe Symptoms:</strong> Provide text description along with images for better analysis</li>
                <li><strong>🎤 Voice Input:</strong> Record your question if typing is inconvenient</li>
                <li><strong>🔍 Get Analysis:</strong> Receive disease diagnosis, pest identification, and treatment advice</li>
            </ul>
        </div>

        <div class="content">
            <div class="input-grid">
                <div class="input-section">
                    <h2>📸 Upload Crop/Plant Image</h2>
                    <div class="image-upload-area" onclick="document.getElementById('imageInput').click()" 
                         ondrop="dropHandler(event)" ondragover="dragOverHandler(event)" 
                         ondragleave="dragLeaveHandler(event)">
                        <div class="upload-icon">📷</div>
                        <p><strong>Click to upload or drag & drop</strong></p>
                        <p style="font-size: 14px; color: #666; margin-top: 10px;">
                            Upload clear photos of affected plants, leaves, or pests
                        </p>
                        <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event)">
                    </div>
                    <div class="image-preview" id="imagePreview"></div>
                </div>

                <div class="input-section">
                    <h2>🌱 Describe Your Problem</h2>
                    <textarea id="question" rows="8" 
                        placeholder="Describe what you're seeing:
- Which crop/plant?
- What symptoms? (spots, wilting, holes, etc.)
- How long has this been happening?
- Weather conditions?
- Any treatments tried?

Example: My tomato plants have yellow spots on leaves, started 3 days ago after heavy rain"></textarea>

                    <div class="button-group">
                        <button class="btn-record" id="recordBtn" onclick="toggleRecording()">🎤 Voice Input</button>
                    </div>
                </div>
            </div>

            <div class="button-group" style="margin-bottom: 20px;">
                <button class="btn-primary" onclick="analyzeWithImage()">🔬 Analyze Image & Get Advice</button>
                <button class="btn-secondary" onclick="clearAll()">🔄 Clear All</button>
            </div>

            <div class="examples">
                <strong>💡 Quick Examples - Click to try:</strong><br>
                <span class="example-btn" onclick="useExample('Yellow spots on tomato leaves')">🍅 Tomato Disease</span>
                <span class="example-btn" onclick="useExample('White insects under cotton leaves')">🐛 Cotton Pest</span>
                <span class="example-btn" onclick="useExample('Brown spots spreading on potato')">🥔 Potato Blight</span>
                <span class="example-btn" onclick="useExample('Yellowing between veins in citrus')">🍊 Nutrient Issue</span>
                <span class="example-btn" onclick="useExample('Holes in corn leaves')">🌽 Corn Borer</span>
                <span class="example-btn" onclick="useExample('Black sooty coating on mango')">🥭 Sooty Mold</span>
            </div>

            <div class="result-section">
                <h2>🔬 Agricultural Analysis & Recommendations</h2>
                <div id="result">
                    <p style="color: #6c757d; text-align: center; padding: 60px;">
                        Upload an image and/or describe your agricultural problem to receive expert analysis.
                    </p>
                </div>
            </div>

            <div class="features">
                <div class="feature-item">
                    <h4>📸 Visual Disease ID</h4>
                    <p>AI analyzes plant images to identify diseases and pests</p>
                </div>
                <div class="feature-item">
                    <h4>🦠 Disease Diagnosis</h4>
                    <p>Get specific disease identification with confidence scores</p>
                </div>
                <div class="feature-item">
                    <h4>💊 Treatment Plans</h4>
                    <p>Receive chemical and organic treatment recommendations</p>
                </div>
                <div class="feature-item">
                    <h4>🛡️ Prevention Tips</h4>
                    <p>Learn how to prevent future occurrences</p>
                </div>
            </div>
        </div>

        <div class="footer">
            <p><strong>Powered by:</strong> TinyLlama + BLIP Vision Model + Agricultural Knowledge Base</p>
            <p style="font-size: 12px; margin-top: 5px;">
                For best results, upload clear, close-up images in good lighting. Consider local conditions and consult agricultural experts for critical decisions.
            </p>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let currentImageBase64 = null;

        function handleImageSelect(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('image/')) {
                displayImage(file);
            }
        }

        function displayImage(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImageBase64 = e.target.result.split(',')[1];
                const preview = document.getElementById('imagePreview');
                preview.innerHTML = `<img src="${e.target.result}" alt="Uploaded image">`;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        function dragOverHandler(event) {
            event.preventDefault();
            event.currentTarget.classList.add('dragover');
        }

        function dragLeaveHandler(event) {
            event.currentTarget.classList.remove('dragover');
        }

        function dropHandler(event) {
            event.preventDefault();
            event.currentTarget.classList.remove('dragover');

            const files = event.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                displayImage(files[0]);
            }
        }

        async function analyzeWithImage() {
            const question = document.getElementById('question').value;

            if (!currentImageBase64 && !question.trim()) {
                alert('Please upload an image or describe your problem!');
                return;
            }

            document.getElementById('result').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p><strong>Analyzing image and symptoms...</strong></p>
                    <p style="font-size: 14px; margin-top: 10px;">This may take 10-15 seconds</p>
                </div>
            `;

            try {
                const response = await fetch('/analyze_agro', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        image: currentImageBase64
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    let resultHTML = '';

                    if (data.image_analysis) {
                        resultHTML += `
                            <div class="image-analysis">
                                <h3>📸 Image Analysis Results</h3>
                                <p>${data.image_analysis}</p>
                            </div>
                        `;
                    }

                    resultHTML += `<div class="advice">${data.advice}</div>`;

                    document.getElementById('result').innerHTML = resultHTML;
                } else {
                    document.getElementById('result').innerHTML = `
                        <p style="color: red; padding: 20px;">❌ Error: ${data.error}</p>
                    `;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `
                    <p style="color: red; padding: 20px;">❌ Error: ${error.message}</p>
                `;
            }
        }

        async function toggleRecording() {
            const btn = document.getElementById('recordBtn');

            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = async () => {
                            const base64Audio = reader.result.split(',')[1];

                            document.getElementById('result').innerHTML = `
                                <div class="loading">
                                    <div class="spinner"></div>
                                    <p>Transcribing audio...</p>
                                </div>
                            `;

                            try {
                                const response = await fetch('/transcribe_agro', {
                                    method: 'POST',
                                    headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({audio: base64Audio})
                                });

                                const data = await response.json();

                                if (response.ok) {
                                    document.getElementById('question').value = data.text;
                                    document.getElementById('result').innerHTML = `
                                        <p style="color: green; padding: 20px;">
                                            ✅ Transcribed! Now upload an image or click "Analyze" to proceed.
                                        </p>
                                    `;
                                } else {
                                    document.getElementById('result').innerHTML = `
                                        <p style="color: red; padding: 20px;">❌ Error: ${data.error}</p>
                                    `;
                                }
                            } catch (error) {
                                document.getElementById('result').innerHTML = `
                                    <p style="color: red; padding: 20px;">❌ Error: ${error.message}</p>
                                `;
                            }
                        };

                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    btn.textContent = '⏹️ Stop Recording';
                    btn.classList.add('recording');
                    isRecording = true;

                } catch (error) {
                    alert('Microphone access denied: ' + error.message);
                }
            } else {
                mediaRecorder.stop();
                btn.textContent = '🎤 Voice Input';
                btn.classList.remove('recording');
                isRecording = false;
            }
        }

        function useExample(text) {
            document.getElementById('question').value = text;
        }

        function clearAll() {
            document.getElementById('question').value = '';
            document.getElementById('imagePreview').style.display = 'none';
            document.getElementById('imagePreview').innerHTML = '';
            document.getElementById('imageInput').value = '';
            currentImageBase64 = null;
            document.getElementById('result').innerHTML = `
                <p style="color: #6c757d; text-align: center; padding: 60px;">
                    Upload an image and/or describe your agricultural problem to receive expert analysis.
                </p>
            `;
        }

        // Keyboard shortcut
        document.getElementById('question').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeWithImage();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML)


@app.route('/analyze_agro', methods=['POST'])
def analyze_agro():
    data = request.json
    question = data.get('question', '')
    image_base64 = data.get('image', '')

    print(f"\n🌾 Agricultural Analysis Request")
    print(f"   Text: {question[:100] if question else 'None'}...")
    print(f"   Image: {'Yes' if image_base64 else 'No'}")

    image_analysis = ""
    combined_context = question

    # If image provided, analyze it first
    if image_base64:
        try:
            print("📸 Analyzing image with BLIP...")

            # Decode image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Save image for records
            timestamp = int(time.time() * 1000)
            image_path = os.path.join(IMAGE_DIR, f"crop_{timestamp}.jpg")
            image.save(image_path)

            # Generate image description
            text_prompt = "This image shows a plant with"
            inputs = blip_processor(image, text_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = blip_model.generate(**inputs, max_new_tokens=50)
                image_description = blip_processor.decode(output[0], skip_special_tokens=True)

            print(f"   BLIP: {image_description}")

            # Additional analysis prompts
            prompts = [
                "The plant disease visible is",
                "The pest damage shows",
                "The leaf condition indicates"
            ]

            analyses = []
            for prompt in prompts:
                inputs = blip_processor(image, prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = blip_model.generate(**inputs, max_new_tokens=30)
                    analysis = blip_processor.decode(output[0], skip_special_tokens=True)
                    if analysis != prompt:  # Only add if model generated something
                        analyses.append(analysis)

            image_analysis = f"""**Visual Analysis:**
• {image_description}
{chr(10).join(['• ' + a for a in analyses if a])}

Based on the image, I can see potential signs of plant stress or disease."""

            # Combine image analysis with text question
            combined_context = f"Image shows: {image_description}. {' '.join(analyses)} User description: {question}" if question else image_description

        except Exception as e:
            print(f"❌ Image analysis error: {e}")
            image_analysis = "**Note:** Could not analyze the image properly. Proceeding with text description."

    # Generate agricultural advice
    prompt = f"""<|system|>
You are an agricultural expert assistant. Analyze the plant problem and provide detailed advice.</s>
<|user|>
{combined_context}

Provide diagnosis and treatment recommendations.</s>
<|assistant|>
"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        print("⏳ Generating agricultural advice...")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|assistant|>" in full_response:
            advice = full_response.split("<|assistant|>")[-1].strip()
        else:
            advice = full_response

        # Clean up
        advice = advice.replace("Chat Doctor", "").replace("http://", "")
        advice = advice[:1500]

        print(f"✅ Generated advice ({len(advice)} chars)\n")

        return jsonify({
            'image_analysis': image_analysis if image_base64 else None,
            'advice': advice
        })

    except Exception as e:
        print(f"❌ Error: {e}\n")
        return jsonify({'error': str(e)}), 500


@app.route('/transcribe_agro', methods=['POST'])
def transcribe_agro():
    data = request.json
    audio_base64 = data.get('audio', '')

    print("\n🎤 Transcription request...")

    if not audio_base64:
        return jsonify({'error': 'No audio data'}), 400

    try:
        # Decode and save audio
        audio_bytes = base64.b64decode(audio_base64)
        timestamp = int(time.time() * 1000)
        audio_filename = os.path.join(AUDIO_DIR, f"agro_{timestamp}.webm")

        with open(audio_filename, 'wb') as f:
            f.write(audio_bytes)

        print(f"   Saved: {audio_filename}")

        # Transcribe
        result = whisper_model.transcribe(audio_filename, fp16=False)
        text = result["text"]

        print(f"✅ Transcribed: {text}\n")

        return jsonify({'text': text})

    except Exception as e:
        print(f"❌ Error: {e}\n")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🌾 AgroExpert Vision System Ready")
    print("📡 Open: http://127.0.0.1:5001")
    print("📸 Image analysis powered by BLIP")
    print("=" * 60 + "\n")

    app.run(host='127.0.0.1', port=5001, debug=False, threaded=False)