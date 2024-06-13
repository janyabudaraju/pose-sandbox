import Dropdown from 'react-bootstrap/Dropdown';
import DropdownButton from 'react-bootstrap/DropdownButton';
import { useState } from "react";

function ModelSelect(){
    const models = [
        { id: 'posenet', name: 'PoseNet' },
        { id: 'blazepose', name: 'BlazePose' },
        { id: 'efficientpose', name: 'EfficientPose' }
    ];

    const [selectedModel, setSelectedModel] = useState(models[0].name);

    return(
        <DropdownButton id="model-select" title={selectedModel}>
            <Dropdown.ItemText>Model Selection</Dropdown.ItemText>
            {models.map(model => (
                <Dropdown.Item key={model.id} onClick={() => setSelectedModel(model.name)}>
                    {model.name}
                </Dropdown.Item>
            ))}
        </DropdownButton>
    )
}

export default ModelSelect;
