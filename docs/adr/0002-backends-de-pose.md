# ADR 0002 — Dois backends de pose: YOLOv8-pose e OpenPose (CMU) em PyTorch

- **Status:** aceito
- **Data:** 2026-10-17

## Contexto

O enunciado cita o **OpenPose** para análise postural e o **YOLOv8** para objetos e áreas críticas. O OpenPose oficial (C++/Caffe) precisa ser compilado com CUDA, o que não roda nos Macs do grupo nem no CI. O caminho alternativo comum, carregar os modelos Caffe pelo módulo DNN do OpenCV, **deixou de existir no OpenCV 5**, que removeu o importador Caffe (`cv2.dnn.readNetFromCaffe` não existe mais na versão instalada).

O pacote `controlnet_aux` roda a rede do OpenPose em PyTorch, mas traz o `opencv-python-headless` (que conflita com o `opencv-python` exigido pelo `ultralytics`, pois os dois instalam o mesmo módulo `cv2`), além de `timm`, `scikit-image` e dos modelos de mão e rosto, que não são usados.

## Decisão

1. Interface única `PoseEstimator.estimate(frame) -> list[Person]`, com keypoints sempre no padrão **COCO-17**. As regras clínicas não sabem qual modelo foi usado.
2. **`YoloPoseEstimator`** (padrão): YOLOv8n-pose, rápido em CPU, usado no dia a dia e na maioria das análises.
3. **`OpenPoseEstimator`**: a rede de corpo original da CMU (COCO-18) **portada para PyTorch e incorporada ao projeto** (`openpose_net.py`, cerca de 300 linhas, com atribuição em `THIRD_PARTY_NOTICES.md`), carregando os pesos oficiais convertidos. Não acrescenta nenhuma dependência além de torch, scipy e opencv, que já estavam no projeto. Roda em CPU, CUDA ou MPS (Apple Silicon).
4. **`read_openpose_json`**: lê a saída JSON (BODY_25) do binário oficial do OpenPose, para quem puder rodá-lo em GPU. Os mapeamentos BODY_25 → COCO-17 e OpenPose-18 → COCO-17 são testados parte a parte.

Ao incorporar o código, corrigimos dois detalhes da origem:
- A última camada de heatmaps (`Mconv7_stage6_L2`) ficou sem ReLU, como no modelo Caffe original. A origem repetia `Mconv7_stage6_L1` na lista de camadas sem ReLU.
- A média entre escalas de inferência foi corrigida (`heatmap_avg += heatmap_avg + ...`).

## Consequências

- Um mesmo vídeo pode ser analisado com os dois modelos (`--backend yolo|openpose`), o que permite compará-los no relatório.
- No mesmo quadro, os dois backends concordam em ~5 px nos keypoints principais (teste `test_real_models.py`).
- O OpenPose é de 5 a 20 vezes mais lento (2–11 quadros/s contra 43–105 do YOLOv8 nesta máquina) e teve recall de quedas menor na validação. Por isso o YOLOv8 é o padrão e o OpenPose fica para comparação e para cumprir o requisito.
- Os pesos do OpenPose têm licença acadêmica não comercial da CMU, adequada ao projeto mas não a uso comercial.
