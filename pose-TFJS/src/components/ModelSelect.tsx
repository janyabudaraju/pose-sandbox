
import { Radio, RadioGroup, Stack, Heading } from "@chakra-ui/react";
// import { PoseModel, BasePose, poseNetModel, moveNetModel, blazeNetModel } from "../utils/ModelDefinitions";
import { modelOptions } from "../utils/ModelDefinitions";

type Props = {
    onModelChange: (modelId: string) => void;
};

function ModelSelect({onModelChange}: Props){
    const handleModelChange = (value: string) => {
        onModelChange(value);
        console.log(value);
    };

    return(
        <div>
            <Heading size="md" mb="2">
                model selection
            </Heading>
            <RadioGroup onChange={handleModelChange} defaultValue={modelOptions[0].id} size="lg">
                <Stack>
                    {
                        modelOptions.map(model => (<Radio key={model.id} value={model.id}> {model.name} </Radio>))
                    }
                </Stack>
            </RadioGroup>
        </div>
    )
}

export default ModelSelect;
