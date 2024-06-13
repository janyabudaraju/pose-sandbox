import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import * as tf from '@tensorflow/tfjs';
import { ChakraProvider } from '@chakra-ui/react'


tf.setBackend('webgl').then(() => {
  console.log("webgl backend initialized");
}).catch(error => {
  console.error("webgl backend failure: ", error);
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ChakraProvider>
      <App />
    </ChakraProvider>
  </React.StrictMode>,
)
