import './App.css'
import ModelSelectCheck from './components/ModelSelectCheck';
import { Box, Flex, Heading } from "@chakra-ui/react";
import { useState } from 'react';
import { modelOptions, PoseModel, BasePose } from './utils/ModelDefinitions';
import WebcamDisplay from './components/WebcamDisplay';
// import VideoUpload from './components/VideoUpload'

function App() {

    // state to handle changes in model selection to pass to analysis component
    const [selectedModels, setSelectedModels] = useState<PoseModel<BasePose>[]>([]);

    // function to handle model change by updating selected models.
    // passed as prop in selection component, state used in analysis component
    const handleModelChange = (modelIds: string[]) => {
        const newModels = modelIds.map(modelId =>
            modelOptions.find(model => model.id === modelId)
        ).filter((model): model is PoseModel<BasePose> => model !== undefined);
    
        setSelectedModels(newModels);
        console.log('set models to:', newModels.map(model => model.id));
    };

    return (
        <Box textAlign="center" p="4">
        <Heading mb="0">pose sandbox</Heading>
        <Flex direction="row" align="center" justify="space-between" w="100%" p="4">
                <Box flex="1" minW="0" w="70%" overflow='auto'>
                    <WebcamDisplay models={selectedModels}/>
                </Box>
                <Box flex="none" minW="200px" borderRadius="lg" p="10px"  m="2" borderWidth="3px" height='95vh'>
                    <ModelSelectCheck onModelChange={handleModelChange}/>
                </Box>
            </Flex>
        </Box>
    );
}
export default App;