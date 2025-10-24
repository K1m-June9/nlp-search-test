from typing import List, Tuple
from konlpy.tag import Komoran
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT

class KeywordExtractor:
    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        stop_words: List[str] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        top_n: int = 10
    ):
        """
        키워드 추출기 초기화
        Args:
            model_name: sentence-transformers 모델 이름
            stop_words: 불용어 리스트
            ngram_range: n-gram 범위
            top_n: 상위 N개 키워드 추출
        """
        self.komoran = Komoran()
        self.model = KeyBERT(model_name)
        self.stop_words = stop_words or ['하는', '있는', '위한', '통한', '되지', '하고']
        self.ngram_range = ngram_range
        self.top_n = top_n
        
        # CountVectorizer 사전 초기화
        self.vectorizer = CountVectorizer(
            tokenizer=self._noun_tokenizer,
            ngram_range=ngram_range,
            max_df=1.0,
            min_df=1
        )

    def _noun_tokenizer(self, text: str) -> List[str]:
        """명사 추출 및 2글자 이상 필터링"""
        try:
            nouns = self.komoran.nouns(text)
            return [noun for noun in nouns if len(noun) >= 2]
        except Exception as e:
            print(f"형태소 분석 중 오류 발생: {e}")
            return []

    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        텍스트에서 키워드 추출
        Args:
            text: 분석할 텍스트
        Returns:
            (키워드, 점수) 형식의 튜플 리스트
        """
        if not text.strip():
            raise ValueError("입력 텍스트가 비어 있습니다.")
            
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            top_n=self.top_n,
            vectorizer=self.vectorizer
        )
        
        return self._postprocess(keywords)

    def _postprocess(self, keywords: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """추출 결과 후처리"""
        # 숫자로만 구성된 키워드 제거
        filtered = [(word, score) for word, score in keywords if not word.isdigit()]
        # 중복 제거 (KeyBERT가 종종 중복을 반환하는 경우 처리)
        seen = set()
        return [(word, score) for word, score in filtered if not (word in seen or seen.add(word))]
    
    def add_stop_words(self, new_stop_words: List[str]):
        """불용어 추가"""
        self.stop_words.extend(new_stop_words)
        # 벡터라이저 재생성
        self.vectorizer = CountVectorizer(
            tokenizer=self._noun_tokenizer,
            ngram_range=self.ngram_range,
            max_df=1.0,
            min_df=1,
            stop_words=self.stop_words
        )