import { Checkbox, CheckboxGroup, Heading, Stack } from '@chakra-ui/react'
import { modelOptions } from "../utils/ModelDefinitions";

type Props = {
    onModelChange: (modelIDs: string[]) => void;
};

function ModelSelectCheck({onModelChange}: Props){

    const handleModelChange = (value: string[]) => {
        onModelChange(value);
        console.log(value);
    };

    return(
        <div>
            <Heading size="md" mb="2">
                model multi-select
            </Heading>
            <CheckboxGroup onChange={handleModelChange} defaultValue={[]} size="lg">
                <Stack>
                    {
                        modelOptions.map(model => (<Checkbox key={model.id} value={model.id}> {model.name} </Checkbox>))
                    }
                </Stack>
            </CheckboxGroup>
        </div>
    )
}

export default ModelSelectCheck;
