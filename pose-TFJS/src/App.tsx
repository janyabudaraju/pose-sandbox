import './App.css'
import WebcamDisplay from './components/WebcamDisplay'
import ModelSelect from './components/ModelSelect';
import { Box, Flex, Heading } from "@chakra-ui/react";

function App() {
    return (
        <Box textAlign="center" p="4"> {/* Container for overall alignment and padding */}
        <Heading mb="1">pose sandbox</Heading> {/* Webpage title */}
        <Flex direction="row" align="center" justify="space-between" w="100%" p="4">
                <Box flex="1" minW="0" w="70%" overflow='auto'> {/* Ensures webcam display can shrink but fills space */}
                <WebcamDisplay />
                </Box>
                <Box flex="none" minW="200px" borderRadius="lg" p="10px"  m="2" borderWidth="3px" height='95vh'> {/* Control panel with fixed minimum width */}
                    <ModelSelect />
                </Box>
            </Flex>
        </Box>
    );
}

export default App;
