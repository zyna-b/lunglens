Pakistan had about 670,000 TB cases in 2024, the 5th highest in the world. Chest X-rays are cheap and everywhere; people who can read them for TB are not.

For my Deep Learning semester project I built LungLens: upload a chest X-ray and, in about a second, it says healthy, sick but not TB, or TB, and shows where in the lungs it looked. It runs entirely in the browser, so the X-ray is never uploaded anywhere.

What I did differently:
• Used TBX11K, not the usual Kaggle TB set, where TB and normal X-rays come from different hospitals (a shortcut a model can learn)
• Three classes, because the hard part is telling TB from other lung disease
• Compared a CNN from scratch with DenseNet-121 starting from ImageNet vs from chest X-rays
• Checked the heatmaps against boxes drawn by radiologists: the hottest point lands inside a box in 36% of TB X-rays (a random point: 10%)

Results on 1,800 test X-rays it never saw: 99.2% accuracy, TB AUC 0.9996, 100.0% sensitivity at 95.9% specificity.

Try it: https://lunglens-tb.vercel.app
Code, weights and data splits: https://github.com/zyna-b/lunglens
Walkthrough video: https://www.youtube.com/watch?v=caT1J9Lhu4A

Thanks to Dr. Munir Ahmad, and to Liu et al. for the TBX11K dataset.
It's a research demo, not a medical device.

#DeepLearning #ComputerVision #HealthcareAI #Tuberculosis #Pakistan #SuperiorUniversity
