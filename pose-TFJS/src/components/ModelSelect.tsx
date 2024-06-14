
import { Radio, RadioGroup, Stack, Heading } from "@chakra-ui/react";
import { PoseModel, Pose2D, poseNetModel } from "../utils/ModelDefinitions";

const modelOptions: (PoseModel<Pose2D>)[] = [ poseNetModel()];

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
                        modelOptions.map(model => (<Radio value={model.name}> {model.name} </Radio>))
                    }
                </Stack>
            </RadioGroup>
        </div>
    )
}

export default ModelSelect;
