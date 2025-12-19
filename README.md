<body>

  <h1>Computer-Aided Detection of Colorectal Polyps</h1>

  <p>
    This repository contains a prototype
    <strong>segmentation-based Computer-Aided Detection (CADe)</strong> system
    designed to automatically detect and segment colorectal polyps in colonoscopy
    images using a deep learning–based <strong>UNet segmentation model</strong>.
  </p>

  <hr />

  <h2>✨ Features</h2>
  <ul>
    <li>Automatic segmentation of polyp regions in colonoscopy frames</li>
    <li>Deep learning–based UNet architecture with ResNet34 encoder</li>
    <li>Support for training, validation, and inference</li>
    <li>Overlay generation: masks applied to original frames</li>
    <li>Cross-dataset evaluation (Train on Kvasir-SEG, Test on CVC-ClinicDB)</li>
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
    <strong>Dataset is NOT included in this repository</strong> due to size constraints.
    After downloading, place the datasets into the following structures
  </p>

  <h3>Kvasir-SEG (Training & Validation)</h3>
  <p>
    The Kvasir-SEG dataset contains 1000 colonoscopy images with ground-truth
    segmentation masks and is used for training and validation.
  </p>

  <p>
    🔗
    <a href="https://datasets.simula.no/kvasir-seg/">
      https://datasets.simula.no/kvasir-seg/
    </a>
  </p>

  <pre>
data/kvasir/
├── images/   (1000 colonoscopy images)
└── masks/    (1000 segmentation masks)
  </pre>

  <h3>CVC-ClinicDB (Cross-Dataset Testing)</h3>
  <p>
    The CVC-ClinicDB dataset is used exclusively for cross-dataset test evaluation
    to assess model generalization on unseen data.
  </p>

  <p>
    🔗
    <a href="https://www.kaggle.com/datasets/balraj98/cvcclinicdb">
      https://www.kaggle.com/datasets/balraj98/cvcclinicdb
    </a>
  </p>

  <pre>
data/cvc_test/
├── images/   (612 colonoscopy images)
└── masks/    (612 segmentation masks)
  </pre>

  <hr />

  <h2>🚀 Getting Started</h2>

  <h3>1️⃣ Install Dependencies</h3>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h3>2️⃣ Prepare Dataset</h3>
  <p>
    Download Kvasir-SEG and CVC-ClinicDB datasets, then place the images and masks
    into the appropriate directories.
  </p>

  <h3>3️⃣ Train the Model (Kvasir-SEG)</h3>
  <pre><code>python train.py</code></pre>
  <p>
    This step performs training and validation using an 80/20 split and computes
    validation loss, Dice, and IoU metrics. The trained model is saved to:
  </p>
  <pre><code>models/unet_polyp.pth</code></pre>

  <h3>4️⃣ Test the Model (CVC-ClinicDB)</h3>
  <pre><code>python test.py</code></pre>
  <p>
    The trained model is evaluated on the unseen CVC-ClinicDB dataset to provide
    final quantitative performance metrics.
  </p>

  <h3>5️⃣ Run Inference (Visualization)</h3>
  <pre><code>python infer.py</code></pre>
  <p>
    This step generates overlay visualizations highlighting predicted polyp regions
    and is intended for qualitative inspection and demonstration.
  </p>

  <hr />

  <h2>📊 Evaluation Metrics</h2>
  <ul>
    <li>Dice Coefficient</li>
    <li>Intersection over Union (IoU)</li>
    <li>Binary Cross-Entropy Loss</li>
  </ul>

  <hr />

  <h2>🧭 Roadmap</h2>
  <ul>
    <li>✅ Trainable UNet segmentation pipeline</li>
    <li>✅ Inference + overlay visualization</li>
    <li>✅ Validation pipeline with Dice and IoU</li>
    <li>✅ Cross-dataset test evaluation (Kvasir → CVC)</li>
    <li>⬜ Sensitivity / recall analysis</li>
    <li>⬜ Best model checkpointing</li>
    <li>⬜ Real-time video segmentation</li>
    <li>⬜ Model optimization and deployment</li>
  </ul>

  <hr />

  <h2>📌 Notes</h2>
  <ul>
    <li>Dataset and trained weights are excluded due to size limitations.</li>
    <li>Model can be retrained with <code>train.py</code>.</li>
    <li>Test data is never used during training or model selection</li>
    <li>
      The project is intended for experimentation and learning purposes rather than
      clinical deployment.
    </li>
  </ul>

</body>

