<body>

  <h1>Computer-Aided Detection of Colorectal Polyps</h1>

  <p>
    This repository contains a prototype <strong>Computer-Aided Detection (CADe)</strong> system 
    designed to automatically detect and segment colorectal polyps in colonoscopy images using 
    a deep learning–based <strong>UNet segmentation model</strong>.
  </p>

  <hr />

  <h2>✨ Features</h2>
  <ul>
    <li>Automatic segmentation of polyp regions in colonoscopy frames</li>
    <li>Deep learning–based UNet architecture with ResNet34 encoder</li>
    <li>Support for training, validation, and inference</li>
    <li>Overlay generation: masks applied to original frames</li>
    <li>Modular and clean code structure (datasets, models, utils, inference)</li>
  </ul>

  <h2>🧱 Tech Stack</h2>
  <ul>
    <li><strong>Language:</strong> Python</li>
    <li><strong>Framework:</strong> PyTorch</li>
    <li><strong>Image Processing:</strong> OpenCV</li>
    <li><strong>Visualization:</strong> Matplotlib / NumPy</li>
  </ul>

  <hr />

  <h2>📂 Dataset</h2>
  <p>
    This project uses the <strong>Kvasir-SEG</strong> dataset, which contains 1000 polyp images and ground-truth masks.
    Download link:
  </p>

  <p>
    🔗 <a href="https://datasets.simula.no/kvasir-seg/">https://datasets.simula.no/kvasir-seg/</a>
  </p>

  <p><strong>Dataset is NOT included in this repository</strong> due to size constraints. 
     After downloading, place the dataset into the following structure:</p>

  <pre>
  data/kvasir/
    ├── images/   (1000 colonoscopy images)
    └── masks/    (1000 segmentation masks)
  </pre>

  <hr />

  <h2>🚀 Getting Started</h2>

  <h3>1️⃣ Install Dependencies</h3>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h3>2️⃣ Prepare Dataset</h3>
  <p>Download Kvasir-SEG and place the images/masks into:</p>
  <pre>data/kvasir/images/
data/kvasir/masks/</pre>

  <h3>3️⃣ Train the Model</h3>
  <pre><code>python train.py</code></pre>

  <p>The trained UNet model will be saved as:</p>
  <pre>models/unet_polyp.pth</pre>

  <h3>4️⃣ Run Inference</h3>
  <pre><code>python infer.py</code></pre>

  <p>
    This will load the model and generate an overlay image highlighting the detected polyp regions.
    The output file:
  </p>
  <pre>overlay_result.png</pre>

  <hr />

  <h2>📊 Evaluation</h2>
  <p>
    The model can be evaluated through common segmentation metrics:
  </p>
  <ul>
    <li>Dice Coefficient</li>
    <li>Intersection over Union (IoU)</li>
    <li>Precision / Recall</li>
    <li>Binary Cross-Entropy Loss</li>
  </ul>

  <hr />

  <h2>🧭 Roadmap</h2>
  <ul>
    <li>✅ Trainable UNet segmentation pipeline</li>
    <li>✅ Inference + overlay visualization</li>
    <li>⬜ Add evaluation metrics and validation pipeline</li>
    <li>⬜ Real-time video segmentation</li>
    <li>⬜ Convert model to TFLite for mobile deployment (Flutter app)</li>
    <li>⬜ Hyperparameter tuning and more advanced architectures</li>
  </ul>

  <hr />

  <h2>📌 Notes</h2>
  <ul>
    <li>Dataset and trained weights are excluded due to size limitations.</li>
    <li>Model can be retrained with <code>train.py</code>.</li>
    <li>The project is structured to make future expansion into mobile apps easy.</li>
  </ul>

</body>
</html>
