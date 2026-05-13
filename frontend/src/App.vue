<template>
  <div class="container">
    <header>
      <h1>Philomath Index</h1>
      <p>AI-Powered Philosophical Density Analyzer</p>
    </header>

    <main>
      <div class="input-section">
        <textarea 
          v-model="inputText" 
          placeholder="Paste at least 30 words of text here (e.g., Kant, Nietzsche, or just a normal email)..."
          rows="8"
        ></textarea>
        <button @click="analyzeText" :disabled="isLoading || inputText.split(' ').length < 30">
          {{ isLoading ? 'Analyzing...' : 'Analyze Text' }}
        </button>
        <p v-if="error" class="error-msg">{{ error }}</p>
      </div>

      <transition name="fade">
        <div v-if="result" class="results-section">
          
          <div class="score-card">
            <h2>Philomath Score</h2>
            <div class="score-circle" :style="getScoreColor(result.score)">
              {{ result.score }}
            </div>
            <p class="stratum-label">{{ getStratumLabel(result.score) }}</p>
          </div>

          <div class="metrics-grid">
            <div class="metric" v-for="(value, key) in result.features_extracted" :key="key">
              <span class="metric-key">{{ getMetricName(key) }}</span>
              <span class="metric-val">{{ value }}</span>
            </div>
          </div>
          
        </div>
      </transition>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const inputText = ref('')
const isLoading = ref(false)
const error = ref(null)
const result = ref(null)

const analyzeText = async () => {
  isLoading.value = true
  error.value = null
  result.value = null

  try {
    const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: inputText.value }),
    })

    if (!response.ok) {
      const errData = await response.json()
      throw new Error(errData.detail || 'Analysis failed')
    }

    result.value = await response.json()
  } catch (e) {
    error.value = e.message
  } finally {
    isLoading.value = false
  }
}

// UI Helpers
const getMetricName = (key) => {
  const names = {
    SDI: 'Syntactic Depth',
    SCD: 'Subclausal Density',
    AR: 'Abstractness Ratio',
    LD: 'Lexical Density',
    VRS: 'Vocabulary Rareness',
    MATTR: 'Lexical Diversity',
    PTD: 'Parse Tree Depth',
    PCD: 'Philosophical Compounds',
    ANR: 'Abstract Noun Repetition',
    PVD: 'Philosophical Vocab Density',
    SLV: 'Sentence Length Variance'
  }
  return names[key] || key
}

const getStratumLabel = (score) => {
  if (score < 20) return "Stratum 1: Foundational (e.g., Aesop)"
  if (score < 40) return "Stratum 2: General Prose (e.g., Twain)"
  if (score < 60) return "Stratum 3: Literary Fiction (e.g., Dickens)"
  if (score < 80) return "Stratum 4: Dense Fiction & Essays (e.g., Kafka)"
  return "Stratum 5: Philosophical Treatises (e.g., Kant/Hegel)"
}

const getScoreColor = (score) => {
  // Gradient from Green (easy) to Red (hard)
  const hue = ((100 - score) * 1.2).toString(10)
  return { border: `6px solid hsl(${hue}, 80%, 50%)`, color: `hsl(${hue}, 80%, 40%)` }
}
</script>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
}

header {
  text-align: center;
  margin-bottom: 3rem;
}

h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: #2c3e50;
}

.input-section textarea {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 1rem;
}

button {
  width: 100%;
  padding: 1rem;
  background-color: #2c3e50;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  cursor: pointer;
  transition: background-color 0.2s;
}

button:disabled {
  background-color: #94a3b8;
  cursor: not-allowed;
}

button:hover:not(:disabled) {
  background-color: #1a252f;
}

.error-msg {
  color: #ef4444;
  margin-top: 1rem;
  text-align: center;
}

.results-section {
  margin-top: 3rem;
  padding: 2rem;
  background: #f8fafc;
  border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}

.score-card {
  text-align: center;
  margin-bottom: 2rem;
}

.score-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 3rem;
  font-weight: bold;
  margin: 1rem auto;
  background: white;
}

.stratum-label {
  font-weight: 600;
  color: #64748b;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.metric {
  background: white;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.metric-key {
  font-size: 0.875rem;
  color: #64748b;
  margin-bottom: 0.5rem;
}

.metric-val {
  font-size: 1.25rem;
  font-weight: bold;
  color: #0f172a;
}

/* Transitions */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.5s ease, transform 0.5s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
  transform: translateY(20px);
}
</style>