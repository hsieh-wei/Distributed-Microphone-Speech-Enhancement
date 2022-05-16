# Overview
This project is the graduation project of HSIEH,WEI and LIN,BO-HAN.The instructor  is FANG,SHIH-HAO. For the detailed content and data arrangement of the project, please refer to
<span style="color:red">
DOC/Distributed Microphone Speech Enhancement.pdf
</span>

# Getting Start(Simulate)
1. You need to record a set of speech that is completely free of ambient noise(We provide some data in "Sample Data")

2. Use <code>Rir-Generator</code> to generate speech files that simulate random walking in space(please save the coordinate)

3. Use <code>Add_Noise</code> to generate noisy datasets

4. Use <code>LA-DDAE to generate</code> enhancement speech

5. Use <code>Evaluation</code> to score your enhancement speech

# Getting Start(Real Field)

1. Set the number of microphones you need for recording in the experimental field, and set up millimeter-wave radar overhead for indoor positioning.

2. Use <code>Positioning</code> to convert the information received by the millimeter wave radar into coordinates.

3. Use <code>Add_Noise</code> to generate noisy datasets

4. Use <code>LA-DDAE to generate</code> enhancement speech

5. Use <code>Evaluation</code> to score your enhancement speech

# Contact Me
If you have ant question of this project, please contact me with wade8954@gmail.com