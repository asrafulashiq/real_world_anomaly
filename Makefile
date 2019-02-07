
train: train-c3d

test: test-c3d

eval: eval-c3d

train-c3d:
	python train.py --mini=false --model-type=c3d \
		--gpus=0,1 \
		--iter=70000

train-c3d-mini:
	python train.py --model-type=c3d \
		--gpus=0,1 --mini=true \
		--iter=20000

train-c3d-attn:
	python train.py --mini=false --model-type=c3d-attn \
		--gpus=0,1 \
		--iter=20000

train-c3d-attn-mini:
	python train.py --model-type=c3d-attn \
		--gpus=0,1 --mini=true \
		--iter=20000


train-3d:
	python train.py --mini=false --model-type=3d \
		--gpus=0,1 \
		--iter=20000 \
		--model-type=3d

train-3d-mini:
	python train.py\
		--gpus=0,1 --mini=true\
		--iter=20000 \
		--model-type=3d


test-c3d:
	python test.py --weight=./model/trained_model/C3D/ \
		--model=./model/trained_model/C3D/model.json \
		--pred=./results/predictions/C3D/ \
		--model-type=c3d

test-c3d-mini:
	python test.py --mini=true --weight=./model/trained_model_mini/C3D/ \
		--model=./model/trained_model_mini/C3D/model.json \
		--pred=./results/predictions/C3D_mini/ \
		--model-type=c3d

test-c3d-attn-mini:
	python test.py --mini=true --weight=./model/trained_model_mini/C3D_attn/ \
		--model=./model/trained_model_mini/C3D_attn/model.json \
		--pred=./results/predictions/C3D_attn_mini/ \
		--model-type=c3d-attn


test-3d:
	python test.py --weight=./model/trained_model/3D/ \
		--model=./model/trained_model/3D/model.json \
		--model-type=3d \
		--pred=./results/predictions/3D/

test-c3d-attn:
	python test.py --weight=./model/trained_model/C3D_attn/ \
		--model=./model/trained_model/C3D_attn/model.json \
		--pred=./results/predictions/C3D_attn/ \
		--model-type=c3d-attn

eval-3d:
	python evaluate.py --pred=./results/predictions/3D/

eval-c3d-attn:
	python evaluate.py --pred=./results/predictions/C3D_attn/

eval-c3d-attn-mini:
	python evaluate.py --pred=./results/predictions/C3D_attn_mini/ --plot --plot-path ./results/plots/C3D_attn/


eval-c3d:
	python evaluate.py --pred=./results/predictions/C3D/ --plot  --plot-path ./results/plots/C3D/

eval-c3d-mini:
	python evaluate.py --pred=./results/predictions/C3D_mini/ --plot  --plot-path ./results/plots/C3D/