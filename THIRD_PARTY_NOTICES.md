# Avisos de terceiros

## OpenPose (rede de corpo)

`src/medwatch/video/pose/openpose_net.py` contém a arquitetura da rede de corpo do OpenPose e o agrupamento de pessoas por Part Affinity Fields, adaptados de:

- **pytorch-openpose**, de Hzzone (https://github.com/Hzzone/pytorch-openpose), licença MIT;
- na versão distribuída pelo **controlnet_aux** (https://github.com/huggingface/controlnet_aux), licença Apache-2.0.

Modelo original: Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh. *Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields*. CVPR 2017. Carnegie Mellon University.

**Pesos:** `body_pose_model.pth` (pesos da CMU convertidos para PyTorch, publicados em https://huggingface.co/lllyasviel/Annotators). São baixados em tempo de execução para `models/` e não são versionados. Os pesos do OpenPose seguem a licença da CMU, **acadêmica e não comercial**, compatível com este projeto acadêmico.

## YOLOv8 (Ultralytics)

Pesos `yolov8n.pt` e `yolov8n-pose.pt`, baixados em tempo de execução pelo pacote `ultralytics` (licença AGPL-3.0).

## Datasets e vídeos

| Recurso | Licença | Crédito |
|---|---|---|
| MIMIC-IV Clinical Database Demo 2.2 | Open Data Commons ODbL 1.0 | Johnson et al., PhysioNet |
| UR Fall Detection Dataset | Uso em pesquisa | B. Kwolek, M. Kepski. *Human fall detection on embedded platform using depth maps and wireless accelerometer*. Computer Methods and Programs in Biomedicine, 2014 |
| `agachamento_unipodal.webm` ("Basic single leg squat") | CC BY-SA 4.0 | RickyBennison, Wikimedia Commons |
| `agachamento_barra.webm` ("Squat - exercise demonstration video") | CC BY 3.0 | FitnessScape, Wikimedia Commons |

Nenhum dataset ou vídeo é redistribuído no repositório. Todos são baixados com `medwatch data download`, que confere o checksum de cada arquivo.
