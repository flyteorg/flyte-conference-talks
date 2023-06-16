# this is a hack, needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	bash setup.sh
	flytectl update task-resource-attribute --attrFile cra.yaml

local-sandbox-setup:
	flytectl demo start --source .
	bash local-setup.sh
	sleep 1
	flytectl update task-resource-attribute --attrFile cra.yaml
