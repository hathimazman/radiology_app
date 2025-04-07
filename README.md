# Radiology AI Learning Platform

A platform for medical students to practice radiology image interpretation with AI-powered feedback.

![Radiology Learning Platform](https://img.shields.io/badge/Medical%20Education-Radiology-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![NLP](https://img.shields.io/badge/NLP-Sentence%20Transformers-yellow)

## Overview

This application helps medical students practice interpreting radiological images by providing:
- Random access to a database of cases with expert annotations
- AI-driven evaluation of student interpretations
- Instant feedback on diagnosis, description, and management plans
- Educational recommendations based on missing key concepts

## Features

- **Interactive Case Study Interface**: View radiological images and submit interpretations
- **AI-Powered Feedback**: Get scored evaluations with specific improvement suggestions
- **Expert Comparison**: Compare your answers with expert interpretations
- **Case Management**: Faculty experts can add new cases to the system

## Science Behind the Model

The platform uses natural language processing (NLP) to evaluate medical students' interpretations:

### Semantic Similarity Analysis

We utilize the `all-MiniLM-L6-v2` model from the sentence-transformers library, which:
- Creates 384-dimensional semantic embeddings that capture the meaning of text
- Compares student and expert answers using cosine similarity
- Understands medical concepts even when phrased differently
- Identifies missing key concepts in student responses

### Why This Approach Works for Medical Education

The transformer-based model understands:
1. **Context-dependent meanings**: Critical for medical terminology
2. **Semantic relationships**: Recognizes synonymous medical terms
3. **Conceptual understanding**: Evaluates whether core medical concepts are present, not just keywords

### Technical Implementation

- **Primary Analysis**: Sentence transformer model for semantic understanding
- **Fallback Mechanism**: NLTK-based keyword analysis when rate limits are hit
- **Robust Caching**: Local model storage to avoid API rate limits
- **Adaptive Feedback**: Domain-specific response generation based on the type of question

## Acknowledgments
- Sentence Transformers library by UKP Lab
- Streamlit for the web application framework
- Supabase for the database backend