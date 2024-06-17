import './App.css'
import WebcamDisplay from './components/WebcamDisplay'
import ModelSelect from './components/ModelSelect';
import { Box, Flex, Heading } from "@chakra-ui/react";
import { useState } from 'react';
import { modelOptions } from './utils/ModelDefinitions';

// TODO:
//  load different models based on dropdown selection
//  have some sort of data logging mechanism. (i could just pull the points, but that feels like missing
//      information.  maybe capture some sort of screen recording also? also need info on the model)
//  figure out the prediction format of each model-- how to convert points to anatomy. maybe add that
//  component (a function) to the model interface
//  start by doing some reading about models that can figure out some of these things-- whether there's a person in frame, whether the person is aligned with camera, etc.
//  then look for labeled datasets. otherwise, i need to figure out a way to actually label the dataset :/

function App() {

    const [selectedModel, setSelectedModel] = useState(modelOptions[0]);
    // console.log(selectedModel);
    const handleModelChange = (modelId: string) => {
        const newModel = modelOptions.find(model => model.id === modelId);
        if (newModel) {
            setSelectedModel(newModel);
            console.log('set model to: ', newModel.id)
        }
        else {
            console.log('invalid selected model');
        }
    };

    return (
        <Box textAlign="center" p="4">
        <Heading mb="0">pose sandbox</Heading>
        <Flex direction="row" align="center" justify="space-between" w="100%" p="4">
                <Box flex="1" minW="0" w="70%" overflow='auto'>
                    <WebcamDisplay model={selectedModel}/>
                </Box>
                <Box flex="none" minW="200px" borderRadius="lg" p="10px"  m="2" borderWidth="3px" height='95vh'>
                    <ModelSelect onModelChange={handleModelChange}/>
                </Box>
            </Flex>
        </Box>
    );
}
export default App;