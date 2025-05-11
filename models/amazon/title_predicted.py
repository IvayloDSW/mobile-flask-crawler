class TitlePredictor:
    def __init__(self, data):
        """
        Initialize the TitlePredictor with data.
        
        Args:
            data (dict): Product data with extracted_text_rows, title, and optionally brand
        """
        self.data = data
        self.extracted_rows = data.get('extracted_text_rows', [])
        self.brand = (data.get("brand") or "").lower()
        self.original_title = data.get('title', '')

    def _clean_text(self, text):
        """
        Clean and normalize text by:
        - Converting to lowercase
        - Removing words shorter than 3 characters
        - Removing standalone numbers
        """
        if not text:
            return ''
            
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Filter out short words (< 3 chars) and standalone numbers
        filtered_words = []
        for word in words:
            # Skip if it's the brand name
            if self.brand and self.brand in word:
                continue
            # Skip if too short
            if len(word) < 3:
                continue
            # Skip if it's just a number
            if word.isdigit():
                continue
            filtered_words.append(word)
            
        return ' '.join(filtered_words)

    def _get_matching_score(self, text, reference):
        """Calculate word match score between cleaned text and reference."""
        clean_text = self._clean_text(text)
        clean_reference = self._clean_text(reference)
        
        if not clean_text or not clean_reference:
            return 0
            
        text_words = set(clean_text.split())
        ref_words = set(clean_reference.split())
        
        if not text_words or not ref_words:
            return 0
            
        # Count matching words
        matching_words = text_words.intersection(ref_words)
        
        if not matching_words:
            return 0
            
        # Calculate score based on number of matching words and their length
        match_ratio = len(matching_words) / max(len(text_words), len(ref_words))
        avg_word_len = sum(len(word) for word in matching_words) / len(matching_words)
        
        # Prioritize longer matches
        return match_ratio * avg_word_len
    
    def _get_brand_free_rows(self):
        """Return rows with brand name filtered out."""
        brand_free_rows = []
        
        for row in self.extracted_rows:
            text = row.get('text', '')
            if not text:
                continue
                
            # Skip rows that are primarily the brand name
            if self.brand and self.brand in text.lower():
                # If brand is significant part of text, skip completely
                if len(self.brand) / len(text.lower()) > 0.4:
                    continue
                    
            # Create a new row with cleaned text
            clean_row = row.copy()
            clean_row['text'] = self._clean_text(text)
            
            # Only add if there's text left after cleaning
            if clean_row['text']:
                brand_free_rows.append(clean_row)
                
        return brand_free_rows
    
    def _find_candidate_rows(self):
        """Find candidate rows for title based on length and match to original title."""
        candidate_rows = self._get_brand_free_rows()
        
        if not candidate_rows:
            return []
            
        # Sort rows by text length (descending)
        candidate_rows.sort(key=lambda r: len(r.get('text', '')), reverse=True)
        
        # Get top longest rows (max 5)
        longest_rows = candidate_rows[:5]
        
        # Calculate match score with original title for each candidate
        scored_rows = []
        for i, row in enumerate(candidate_rows):
            score = self._get_matching_score(row.get('text', ''), self.original_title)
            scored_rows.append((i, row, score))
        
        # Sort by score (descending)
        scored_rows.sort(key=lambda x: x[2], reverse=True)
        
        # No good matches
        if not scored_rows or scored_rows[0][2] == 0:
            return []
            
        # Start with best matching row
        best_idx, best_row, _ = scored_rows[0]
        selected_rows = [best_row]
        
        # Check adjacent rows (previous and next only)
        for adj_idx in [best_idx-1, best_idx+1]:
            if 0 <= adj_idx < len(candidate_rows):
                adj_row = candidate_rows[adj_idx]
                # Only include if it matches original title
                if self._get_matching_score(adj_row.get('text', ''), self.original_title) > 0:
                    selected_rows.append(adj_row)
        
        return selected_rows
    
    def _combine_row_metrics(self, rows):
        """Combine metrics from multiple rows."""
        if not rows:
            return {}
            
        if len(rows) == 1:
            return rows[0]
            
        # Combine text with spaces
        combined_text = ' '.join(row.get('text', '') for row in rows)
        
        # Calculate combined height
        total_height = sum(row.get('height', 0) for row in rows)
        
        # Use metrics from row with highest contrast ratio for colors
        best_contrast_row = max(rows, key=lambda r: r.get('contrast_ratio', 0))
        
        return {
            'text': combined_text,
            'height': total_height,
            'background_color_rgb': best_contrast_row.get('background_color_rgb', [240, 240, 240]),
            'background_color_hex': best_contrast_row.get('background_color_hex', '#f0f0f0'),
            'text_color_rgb': best_contrast_row.get('text_color_rgb', [0, 0, 0]),
            'text_color_hex': best_contrast_row.get('text_color_hex', '#000000'),
            'contrast_ratio': best_contrast_row.get('contrast_ratio', 1.0),
            'visible': any(row.get('visible', False) for row in rows),
            'position_percentage': rows[0].get('position_percentage', 0)
        }
    
    def predict_title(self):
        """Predict the product title from extracted text rows."""
        # Find rows that are candidates for the title
        title_rows = self._find_candidate_rows()
        
        if not title_rows:
            # Fallback to longest row
            filtered_rows = self._get_brand_free_rows()
            if filtered_rows:
                longest_row = max(filtered_rows, key=lambda r: len(r.get('text', '')))
                return {
                    'predicted_title': longest_row.get('text', ''),
                    'confidence': 'low',
                    'method': 'longest_text_fallback',
                    'metrics': longest_row
                }
            return {
                'predicted_title': '',
                'confidence': 'none',
                'method': 'no_valid_text_found',
                'metrics': {}
            }
        
        # Combine metrics from selected rows
        combined_metrics = self._combine_row_metrics(title_rows)
        predicted_title = combined_metrics.get('text', '')
        
        # Calculate match score with original title
        match_score = self._get_matching_score(predicted_title, self.original_title)
        
        # Determine confidence level
        if match_score > 15:
            confidence = 'very_high'
        elif match_score > 10:
            confidence = 'high'
        elif match_score > 7:
            confidence = 'medium'
        elif match_score > 4:
            confidence = 'low'
        else:
            confidence = 'very_low'
            
        return {
            'predicted_title': predicted_title,
            'confidence': confidence,
            'method': 'text_matching',
            'match_score': round(match_score, 2),
            'metrics': combined_metrics
        }