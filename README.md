# Cloud Camera Processing

The project implements a Smart Lighting System in which public webcams act as presence sensors. Frames are periodically captured from fixed cameras and analysed to identify people and vehicles.

The workflow includes:

1. local capture and local YOLOv8 inference;
2. frame capture on the local computer and inference on an Azure virtual machine;
3. serverless processing in Azure;
4. extension of both cloud architectures to multiple cameras.

The repository contains the local single-camera implementation, the multi-camera VM and serverless implementations, the experiments for one, two, and three cameras, and the VM stress tests for 5, 10, and 15 virtual cameras. Saved `.webarchive` files provide visual examples of the dashboards without running the application.

Run commands are provided in each implementation directory. The complete description and results are available in `[Practice1_report.pdf](https://github.com/tomabes/cloud-camera-processing/blob/main/Tamara_Besedina_Cloud_Report.pdf)`.
