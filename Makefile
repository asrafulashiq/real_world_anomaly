
train: train-c3d

test: test-c3d

eval: eval-c3d

train-c3d:
	python train.py --c3d=true \
		--gpus=0,1 \
		--iter=20000

train-c3d-mini:
	python train.py --c3d=true \
		--gpus=0,1 --mini=true \
		--iter=20000

test-c3d:
	python test.py --weight=./model/trained_model/C3D/ \
		--model=./model/trained_model/C3D/model.json \
		--pred=./results/predictions/C3D/

test-c3d-mini:
	python test.py --mini=true --weight=./model/trained_model/C3D/ \
		--model=./model/trained_model/C3D/model.json \
		--pred=./results/predictions/C3D/

eval-c3d:
	python evaluate.py --pred=./results/predictions/C3D/

train-3d:
	python train.py --c3d=false \
		--gpus=0,1 \
		--iter=20000

test-3d:
	python test.py --weight=./model/trained_model/3D/ \
		--model=./model/trained_model/3D/model.json \
		--pred=./results/predictions/3D/

eval-3d:
	python evaluate.py --pred=./results/predictions/3D/