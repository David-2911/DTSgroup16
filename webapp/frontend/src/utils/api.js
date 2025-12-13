/**
 * API utility for backend communication
 */
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

/**
 * Classify uploaded image
 * @param {File} imageFile - The MRI image file
 * @returns {Promise} - Classification results
 */
export const classifyImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  try {
    const response = await axios.post(`${API_BASE_URL}/classify`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000 // 60 second timeout
    });
    
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error
      throw new Error(error.response.data.error || 'Classification failed');
    } else if (error.request) {
      // Request made but no response
      throw new Error('No response from server. Please check if backend is running.');
    } else {
      // Something else went wrong
      throw new Error('Error uploading image: ' + error.message);
    }
  }
};

/**
 * Check API health
 * @returns {Promise} - Health status
 */
export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`, {
      timeout: 5000
    });
    return response.data;
  } catch (error) {
    throw new Error('Backend server is not responding');
  }
};
