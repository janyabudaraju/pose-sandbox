
import { useState } from "react";
import { Radio, RadioGroup, Stack, Heading } from "@chakra-ui/react";

function ModelSelect(){
    const models = [
        { id: 'posenet', name: 'PoseNet' },
        { id: 'blazepose', name: 'BlazePose' },
        { id: 'efficientpose', name: 'EfficientPose' }
    ];
    const [selectedModel, setSelectedModel] = useState(models[0].name);

    return(
        <div>
            <Heading size="md" mb="2">
                model selection
            </Heading>
            <RadioGroup onChange={setSelectedModel} value={selectedModel} size="lg">
                <Stack>
                    {
                        models.map(model => (<Radio value={model.name}> {model.name} </Radio>))
                    }
                </Stack>
            </RadioGroup>
        </div>
    )
}

export default ModelSelect;
