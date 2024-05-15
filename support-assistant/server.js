const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const port = 30080;

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.post('/support', async (req, res) => {
  try {
    const { email, issue, language } = req.body;
    console.log('Received issue:', issue);

    const { handleSupportRequest } = await import('./assistant-core.mjs');
    await handleSupportRequest(email, issue, language);

    res.status(200).send('Support request received and processed');
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'An error occurred while processing the support request.' });
  }
});

app.use((err, req, res, next) => {
  console.error('Global error:', err.stack);
  res.status(500).json({ error: 'An unexpected error occurred.' });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});