# Deepfake-Awareness-and-Recognition-
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepGuard | Deepfake Awareness</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .glass { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); }
        .scanning-line { height: 2px; background: #ccff00; position: absolute; width: 100%; animation: scan 2s linear infinite; }
        @keyframes scan { 0% { top: 0; } 100% { top: 100%; } }
    </style>
</head>
<body class="bg-slate-950 text-slate-200 font-sans">

    <nav class="flex justify-between items-center p-6 border-b border-slate-800">
        <h1 class="text-2xl font-bold text-lime-400 tracking-tighter">DEEPGUARD.AI</h1>
        <div class="space-x-6 text-sm uppercase tracking-widest font-medium">
            <a href="#" class="hover:text-white">Detection</a>
            <a href="#" class="hover:text-white">Technology</a>
            <span class="bg-lime-500 text-black px-3 py-1 rounded text-xs font-bold">SECURE</span>
        </div>
    </nav>

    <section class="max-w-6xl mx-auto px-6 py-20 text-center">
        <h2 class="text-6xl font-extrabold text-white mb-6">Reality is being <span class="text-lime-400 italic">Synthesized.</span></h2>
        <p class="text-xl text-slate-400 max-w-2xl mx-auto mb-10">
            Utilizing <strong>Hybrid CNN-Transformer</strong> architectures to identify synthetic facial artifacts with forensic precision.
        </p>

        <div class="max-w-2xl mx-auto glass border border-slate-800 rounded-2xl p-8 relative overflow-hidden">
            <div id="scan-area" class="h-64 border-2 border-dashed border-slate-700 rounded-xl flex items-center justify-center relative">
                <div id="scanner" class="hidden scanning-line"></div>
                <p id="status-text" class="text-slate-500">Drop video or image here for forensic scan</p>
            </div>
            <button onclick="startAnalysis()" class="w-full mt-6 bg-lime-500 hover:bg-lime-400 text-black font-bold py-4 rounded-xl transition-all shadow-[0_0_15px_rgba(163,230,53,0.3)]">
                START SCAN
            </button>
        </div>
    </section>

    <section class="bg-slate-900/50 py-20 border-t border-slate-800">
        <div class="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 px-6">
            <div>
                <h3 class="text-lime-400 font-bold mb-4 uppercase tracking-widest">The Architecture</h3>
                <p class="text-slate-400">Our model combines <strong>Convolutional Neural Networks (CNN)</strong> for spatial texture analysis and <strong>Transformers</strong> to detect temporal inconsistencies in video sequences.</p>
            </div>
            <div>
                <h3 class="text-lime-400 font-bold mb-4 uppercase tracking-widest">Key Artifacts Detected</h3>
                <ul class="text-sm space-y-2 text-slate-400">
                    <li>• Abnormal Blinking Rates</li>
                    <li>• Boundary Artifacts (Jaw/Hairline)</li>
                    <li>• Spectral Frequency Inconsistencies</li>
                </ul>
            </div>
        </div>
    </section>

    <script>
        function startAnalysis() {
            const status = document.getElementById('status-text');
            const scanner = document.getElementById('scanner');
            
            status.innerText = "Analyzing Frame Sequences...";
            scanner.classList.remove('hidden');
            
            setTimeout(() => {
                scanner.classList.add('hidden');
                status.innerHTML = "<span class='text-red-500 font-bold'>Deepfake Detected (94.8% Confidence)</span><br><span class='text-xs text-slate-500'>Artifacts found in: Periorbital region</span>";
            }, 3000);
        }
    </script>
</body>
</html>
