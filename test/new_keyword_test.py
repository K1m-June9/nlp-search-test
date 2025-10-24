# 한국어 요약 데이터셋 기반 키워드 추출 모델 비교 평가 시스템
# KoreanSummarizeAiHub 데이터셋 사용

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
import random
import re
from collections import Counter
from tqdm import tqdm
import subprocess
import sys

warnings.filterwarnings('ignore')

def install_requirements():
    """필요한 패키지 설치"""
    packages = [
        'datasets',
        'keybert', 
        'sentence-transformers',
        'konlpy',
        'networkx',
        'scikit-learn',
        'nltk',
        'scipy'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 설치 완료")
        except:
            print(f"✗ {package} 설치 실패")

# 패키지 설치
print("필요한 패키지 설치 중...")
install_requirements()

# 라이브러리 임포트
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import networkx as nx
from scipy import stats

# KoNLPy 임포트 (에러 처리 포함)
try:
    from konlpy.tag import Okt, Komoran
    KONLPY_AVAILABLE = True
    print("✓ KoNLPy 사용 가능")
except:
    KONLPY_AVAILABLE = False
    print("⚠️ KoNLPy 사용 불가 - 기본 토크나이저 사용")

class KoreanKeywordDatasetConverter:
    """한국어 요약 데이터셋을 키워드 추출용으로 변환"""
    
    def __init__(self):
        if KONLPY_AVAILABLE:
            try:
                self.okt = Okt()
                self.use_advanced_tokenizer = True
                print("✓ 고급 한국어 토크나이저 사용")
            except:
                self.use_advanced_tokenizer = False
                print("⚠️ 기본 토크나이저로 대체")
        else:
            self.use_advanced_tokenizer = False
        
        # 한국어 불용어
        self.stopwords = set([
            '이', '그', '저', '것', '수', '등', '들', '및', '또한', '하지만', '그러나',
            '따라서', '그래서', '또는', '그리고', '하는', '되는', '있는', '없는',
            '이런', '그런', '저런', '같은', '다른', '새로운', '많은', '적은', '위한',
            '통해', '대한', '관한', '에서', '에게', '으로', '로서', '부터', '까지'
        ])
        
    def extract_nouns_from_text(self, text):
        """텍스트에서 명사 추출"""
        if not text:
            return []
            
        try:
            if self.use_advanced_tokenizer:
                # KoNLPy 사용
                pos_tags = self.okt.pos(text, stem=True)
                nouns = [word for word, pos in pos_tags 
                        if pos in ['Noun'] and len(word) >= 2]
            else:
                # 기본 정규식 사용
                nouns = re.findall(r'[가-힣]{2,}', text)
            
            # 불용어 제거 및 필터링
            filtered_nouns = [noun for noun in nouns 
                            if noun not in self.stopwords 
                            and len(noun) >= 2 
                            and not noun.isdigit()
                            and len(noun) <= 10]  # 너무 긴 단어 제외
            
            return filtered_nouns
            
        except Exception as e:
            print(f"⚠️ 명사 추출 오류: {e}")
            # 대체 방법
            words = re.findall(r'[가-힣]{2,}', text)
            return [w for w in words if w not in self.stopwords]
    
    def calculate_keyword_importance(self, nouns, text):
        """명사들의 중요도 계산"""
        if not nouns:
            return {}
            
        noun_counts = Counter(nouns)
        text_length = len(text)
        total_nouns = len(nouns)
        
        importance_scores = {}
        
        for noun, count in noun_counts.items():
            # 1. TF 점수 (빈도)
            tf_score = count / total_nouns
            
            # 2. 길이 보너스 (긴 명사일수록 구체적)
            length_bonus = min(len(noun) / 8, 0.5)
            
            # 3. 위치 보너스 (앞쪽에 나올수록 중요)
            try:
                first_position = text.find(noun) / text_length
                position_bonus = (1 - first_position) * 0.3
            except:
                position_bonus = 0
            
            # 4. 빈도 보너스 (적당한 빈도가 좋음)
            frequency_bonus = min(count / 3, 0.3) if count > 1 else 0
            
            # 총 점수 계산
            total_score = tf_score + length_bonus + position_bonus + frequency_bonus
            importance_scores[noun] = total_score
            
        return importance_scores
    
    def extract_reference_keywords(self, passage, summary, method='summary_based'):
        """참조 키워드 추출"""
        
        if method == 'summary_based':
            # 요약문 기반 키워드 추출
            summary_nouns = self.extract_nouns_from_text(summary)
            importance_scores = self.calculate_keyword_importance(summary_nouns, summary)
            
        elif method == 'comparison_based':
            # 원문-요약문 비교 기반
            passage_nouns = set(self.extract_nouns_from_text(passage))
            summary_nouns = self.extract_nouns_from_text(summary)
            
            # 요약문에 있으면서 원문에도 있는 명사들 (중요한 키워드일 가능성 높음)
            common_nouns = [noun for noun in summary_nouns if noun in passage_nouns]
            
            if common_nouns:
                importance_scores = self.calculate_keyword_importance(common_nouns, summary)
            else:
                # 공통 명사가 없으면 요약문 기반으로 대체
                importance_scores = self.calculate_keyword_importance(summary_nouns, summary)
                
        elif method == 'hybrid':
            # 하이브리드 방법
            summary_nouns = self.extract_nouns_from_text(summary)
            passage_nouns = set(self.extract_nouns_from_text(passage))
            
            # 요약문 명사들의 중요도 계산
            summary_importance = self.calculate_keyword_importance(summary_nouns, summary)
            
            # 원문에도 있는 명사들에게 보너스 점수
            importance_scores = {}
            for noun, score in summary_importance.items():
                bonus = 0.5 if noun in passage_nouns else 0
                importance_scores[noun] = score + bonus
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 중요도 순으로 정렬
        if not importance_scores:
            return []
            
        sorted_keywords = sorted(importance_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # 상위 키워드 반환 (중복 제거)
        keywords = []
        seen = set()
        for keyword, score in sorted_keywords:
            if keyword.lower() not in seen and len(keywords) < 15:  # 더 많이 생성
                keywords.append(keyword)
                seen.add(keyword.lower())
                
        return keywords
    
    def convert_dataset(self, dataset, num_samples=100, keyword_method='summary_based'):
        """데이터셋 변환"""
        print(f"\n📊 데이터셋 변환 중... (방법: {keyword_method})")
        
        converted_data = []
        successful_conversions = 0
        
        for i, item in enumerate(dataset[:num_samples * 2]):  # 여유있게 더 많이 시도
            try:
                passage = item['passage']
                summary = item['summary']
                
                # 텍스트 품질 확인
                if len(passage) < 50 or len(summary) < 10:
                    continue
                
                # 참조 키워드 생성
                reference_keywords = self.extract_reference_keywords(
                    passage, summary, method=keyword_method
                )
                
                # 키워드가 충분히 생성되었는지 확인
                if len(reference_keywords) >= 3:
                    # 원문에서 키워드 추출 테스트
                    converted_data.append({
                        'text': passage,
                        'keywords': reference_keywords,
                        'summary': summary,
                        'data_type': 'passage',
                        'original_index': i
                    })
                    
                    # 요약문에서 키워드 추출 테스트
                    converted_data.append({
                        'text': summary,
                        'keywords': reference_keywords,
                        'passage': passage,
                        'data_type': 'summary', 
                        'original_index': i
                    })
                    
                    successful_conversions += 1
                    
                    if (successful_conversions) % 10 == 0:
                        print(f"  - {successful_conversions}개 변환 완료")
                        print(f"    예시 키워드: {reference_keywords[:5]}")
                
                # 목표 샘플 수에 도달하면 중단
                if successful_conversions >= num_samples:
                    break
                    
            except Exception as e:
                print(f"⚠️ 샘플 {i} 변환 오류: {e}")
                continue
        
        print(f"✅ 총 {len(converted_data)}개 데이터 변환 완료 (원본 {successful_conversions}개)")
        return converted_data

class KoreanKeywordExtractorComparison:
    """한국어 키워드 추출 비교 평가 클래스"""
    
    def __init__(self, num_samples=50, random_seed=42):
        self.num_samples = num_samples
        self.random_seed = random_seed
        
        # 랜덤 시드 설정
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 결과 저장용
        self.results = {}
        self.predictions = {}
        self.references = []
        self.test_texts = []
        
        print(f"🚀 한국어 키워드 추출 비교 시스템 초기화 (시드: {random_seed})")
        
    def load_models(self):
        """모든 모델 로드"""
        print("\n📥 모델 로딩 중...")
        
        # KeyBERT (한국어 특화)
        print("  - KeyBERT 로딩...")
        try:
            self.keybert = KeyBERT('klue/bert-base')
            print("    ✓ KLUE BERT 모델 사용")
        except:
            self.keybert = KeyBERT('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("    ✓ Multilingual 모델 사용")
        
        # 의미적 유사도 계산용
        print("  - Sentence-BERT 로딩...")
        try:
            self.semantic_model = SentenceTransformer('klue/bert-base')
        except:
            self.semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # 한국어 토크나이저
        if KONLPY_AVAILABLE:
            try:
                self.okt = Okt()
                self.use_advanced_tokenizer = True
            except:
                self.use_advanced_tokenizer = False
        else:
            self.use_advanced_tokenizer = False
        
        # 한국어 불용어
        self.korean_stopwords = set([
            '이', '그', '저', '것', '수', '등', '들', '및', '또한', '하지만', '그러나',
            '따라서', '그래서', '또는', '그리고', '하는', '되는', '있는', '없는',
            '이런', '그런', '저런', '같은', '다른', '새로운', '많은', '적은'
        ])
        
        # TF-IDF는 나중에 한국어 데이터로 학습
        self.tfidf_vectorizer = None
        
        print("✅ 모델 로딩 완료!")
        
    def _korean_tokenizer(self, text):
        """한국어 토크나이저"""
        if not text:
            return []
            
        try:
            if self.use_advanced_tokenizer:
                pos_tags = self.okt.pos(text, stem=True)
                keywords = [word for word, pos in pos_tags 
                           if pos in ['Noun', 'Adjective'] and len(word) >= 2]
            else:
                keywords = re.findall(r'[가-힣]{2,}', text)
            
            return [kw for kw in keywords if kw not in self.korean_stopwords]
            
        except:
            words = re.findall(r'[가-힣]{2,}', text)
            return [w for w in words if w not in self.korean_stopwords]
    
    def _preprocess_text(self, text):
        """텍스트 전처리"""
        if not text:
            return ""
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _post_process_keywords(self, keywords, top_k=10):
        """키워드 후처리"""
        if not keywords:
            return []
        
        # 튜플에서 키워드만 추출
        processed = []
        for kw in keywords:
            if isinstance(kw, tuple):
                processed.append(str(kw[0]))
            else:
                processed.append(str(kw))
        
        # 필터링
        filtered = []
        for kw in processed:
            kw = kw.strip()
            if 2 <= len(kw) <= 15 and not kw.isdigit():
                if re.search(r'[가-힣]', kw):  # 한국어 포함
                    filtered.append(kw)
        
        # 중복 제거
        seen = set()
        unique_keywords = []
        for kw in filtered:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords[:top_k]
    
    def load_korean_dataset(self):
        """한국어 요약 데이터셋 로드 및 변환"""
        print(f"\n📊 한국어 데이터셋 로딩 중...")
        
        try:
            # 한국어 요약 데이터셋 로드
            dataset = load_dataset("Laplace04/KoreanSummarizeAiHub")
            test_data = dataset['test']
            print(f"원본 데이터셋 크기: {len(test_data)}")
            
            # 키워드 추출용으로 변환
            converter = KoreanKeywordDatasetConverter()
            
            # 여러 방법으로 변환 (가장 좋은 방법 선택)
            print("\n다양한 키워드 생성 방법 테스트 중...")
            
            methods = ['summary_based', 'comparison_based', 'hybrid']
            best_method = 'hybrid'  # 기본값
            best_quality = 0
            
            for method in methods:
                print(f"\n--- {method} 방법 테스트 ---")
                sample_data = converter.convert_dataset(
                    test_data, num_samples=10, keyword_method=method
                )
                
                # 품질 평가 (키워드 개수와 다양성)
                if sample_data:
                    avg_keywords = np.mean([len(item['keywords']) for item in sample_data])
                    keyword_diversity = len(set([kw for item in sample_data 
                                               for kw in item['keywords']])) / max(1, len([kw for item in sample_data for kw in item['keywords']]))
                    quality_score = avg_keywords * keyword_diversity
                    
                    print(f"  평균 키워드 수: {avg_keywords:.1f}")
                    print(f"  키워드 다양성: {keyword_diversity:.3f}")
                    print(f"  품질 점수: {quality_score:.3f}")
                    
                    if quality_score > best_quality:
                        best_quality = quality_score
                        best_method = method
            
            print(f"\n✅ 최적 방법 선택: {best_method}")
            
            # 최적 방법으로 전체 데이터 변환
            converted_data = converter.convert_dataset(
                test_data, num_samples=self.num_samples, keyword_method=best_method
            )
            
            # 원문과 요약문 데이터 분리
            passage_data = [item for item in converted_data if item['data_type'] == 'passage']
            summary_data = [item for item in converted_data if item['data_type'] == 'summary']
            
            # 요약문 데이터 사용 (키워드 추출에 더 적합)
            self.test_data = summary_data[:self.num_samples]
            self.test_texts = [item['text'] for item in self.test_data]
            self.references = [item['keywords'] for item in self.test_data]
            
            print(f"✅ 최종 테스트 데이터: {len(self.test_data)}개")
            
            # 샘플 출력
            if self.test_data:
                print(f"\n📝 샘플 예시:")
                sample = self.test_data[0]
                print(f"텍스트: {sample['text'][:100]}...")
                print(f"키워드: {sample['keywords'][:5]}")
            
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {e}")
            self.test_data = []
    
    def prepare_korean_tfidf(self):
        """한국어 TF-IDF 준비"""
        print("\n⚖️ 한국어 TF-IDF 준비 중...")
        
        try:
            # 현재 테스트 데이터의 일부를 학습용으로 사용 (교차 검증 방식)
            if len(self.test_data) > 20:
                # 앞의 20%는 TF-IDF 학습용, 나머지는 테스트용
                split_idx = len(self.test_data) // 5
                train_texts = [item['text'] for item in self.test_data[:split_idx]]
                
                # 추가로 요약문들도 학습 데이터에 포함
                if 'passage' in self.test_data[0]:
                    train_texts.extend([item['passage'] for item in self.test_data[:split_idx] 
                                      if 'passage' in item])
                
                print(f"  TF-IDF 학습 데이터: {len(train_texts)}개")
            else:
                # 데이터가 적으면 전체 텍스트로 학습
                train_texts = self.test_texts
                print(f"  TF-IDF 학습 데이터: {len(train_texts)}개 (전체)")
            
            # 한국어 특화 TF-IDF 벡터라이저
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=self._korean_tokenizer,
                ngram_range=(1, 2),
                max_features=5000,
                min_df=1,
                max_df=0.9,
                lowercase=False,  # 한국어는 대소문자 구분 없음
                sublinear_tf=True
            )
            
            # 한국어 텍스트로 학습
            self.tfidf_vectorizer.fit(train_texts)
            print("✅ 한국어 TF-IDF 학습 완료!")
            
        except Exception as e:
            print(f"⚠️ TF-IDF 준비 오류: {e}")
            # 최소한의 벡터라이저
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),
                max_features=1000,
                min_df=1
            )
            basic_texts = ["한국어 키워드 추출 테스트"] * 5
            self.tfidf_vectorizer.fit(basic_texts)
            print("✅ 기본 TF-IDF 학습 완료!")
    
    def extract_keybert_keywords(self, text, top_k=10):
        """KeyBERT 키워드 추출"""
        try:
            if not text.strip():
                return []
            
            keywords = self.keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words=None,
                top_n=top_k * 2,
                use_mmr=True,
                diversity=0.5,
                use_maxsum=True,
                nr_candidates=20
            )
            
            keyword_list = [kw[0] for kw in keywords]
            return self._post_process_keywords(keyword_list, top_k)
            
        except Exception as e:
            print(f"⚠️ KeyBERT 오류: {e}")
            return []
    
    def extract_tfidf_keywords(self, text, top_k=10):
        """TF-IDF 키워드 추출"""
        try:
            if not text.strip() or self.tfidf_vectorizer is None:
                return []
            
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_vector.toarray()[0]
            
            # 상위 키워드 추출
            top_indices = scores.argsort()[-(top_k * 2):][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return self._post_process_keywords(keywords, top_k)
            
        except Exception as e:
            print(f"⚠️ TF-IDF 오류: {e}")
            return []
    
    def extract_textrank_keywords(self, text, top_k=10):
        """TextRank 키워드 추출"""
        try:
            if not text.strip():
                return []
            
            tokens = self._korean_tokenizer(text)
            
            if len(tokens) < 3:
                return self._post_process_keywords(tokens, top_k)
            
            # 그래프 생성
            graph = nx.Graph()
            window_size = min(5, max(3, len(tokens) // 8))
            
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i:i + window_size]
                for j in range(len(window)):
                    for k in range(j + 1, len(window)):
                        if window[j] != window[k]:
                            distance_weight = 1.0 / (abs(j - k) + 1)
                            
                            if graph.has_edge(window[j], window[k]):
                                graph[window[j]][window[k]]['weight'] += distance_weight
                            else:
                                graph.add_edge(window[j], window[k], weight=distance_weight)
            
            if len(graph.nodes()) == 0:
                return self._post_process_keywords(tokens, top_k)
            
            pagerank_scores = nx.pagerank(graph, weight='weight', alpha=0.85)
            sorted_keywords = sorted(pagerank_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            keywords = [kw[0] for kw in sorted_keywords]
            return self._post_process_keywords(keywords, top_k)
            
        except Exception as e:
            print(f"⚠️ TextRank 오류: {e}")
            return []
    
    def extract_hybrid_keywords(self, text, top_k=10):
        """하이브리드 키워드 추출"""
        try:
            if not text.strip():
                return []
            
            # 각 방법으로 키워드 추출
            textrank_keywords = self.extract_textrank_keywords(text, top_k * 2)
            keybert_keywords = self.extract_keybert_keywords(text, top_k * 2)
            
            # 키워드 점수 계산
            keyword_scores = {}
            
            # TextRank 점수 (구조적 중요도)
            for i, kw in enumerate(textrank_keywords):
                score = (len(textrank_keywords) - i) / len(textrank_keywords) if textrank_keywords else 0
                keyword_scores[kw] = keyword_scores.get(kw, 0) + score * 0.4
            
            # KeyBERT 점수 (의미적 중요도)
            for i, kw in enumerate(keybert_keywords):
                score = (len(keybert_keywords) - i) / len(keybert_keywords) if keybert_keywords else 0
                keyword_scores[kw] = keyword_scores.get(kw, 0) + score * 0.6
            
            # 공통 키워드 보너스
            common_keywords = set(textrank_keywords) & set(keybert_keywords)
            for kw in common_keywords:
                keyword_scores[kw] = keyword_scores.get(kw, 0) + 0.3
            
            # 점수 기반 정렬
            if keyword_scores:
                sorted_keywords = sorted(keyword_scores.items(), 
                                       key=lambda x: x[1], reverse=True)
                hybrid_keywords = [kw[0] for kw in sorted_keywords]
                return self._post_process_keywords(hybrid_keywords, top_k)
            else:
                return []
                
        except Exception as e:
            print(f"⚠️ Hybrid 오류: {e}")
            return self.extract_keybert_keywords(text, top_k)
    
    def extract_all_keywords(self):
        """모든 방법으로 키워드 추출"""
        print(f"\n🔄 키워드 추출 시작 ({len(self.test_data)}개 샘플)...")
        
        keybert_predictions = []
        tfidf_predictions = []
        textrank_predictions = []
        hybrid_predictions = []
        
        for i, item in enumerate(tqdm(self.test_data, desc="키워드 추출")):
            text = item['text']
            
            # 각 방법으로 키워드 추출
            keybert_kw = self.extract_keybert_keywords(text, top_k=10)
            tfidf_kw = self.extract_tfidf_keywords(text, top_k=10)
            textrank_kw = self.extract_textrank_keywords(text, top_k=10)
            hybrid_kw = self.extract_hybrid_keywords(text, top_k=10)
            
            keybert_predictions.append(keybert_kw)
            tfidf_predictions.append(tfidf_kw)
            textrank_predictions.append(textrank_kw)
            hybrid_predictions.append(hybrid_kw)
            
            # 진행상황 출력
            if (i + 1) % 10 == 0:
                print(f"\n--- 샘플 {i+1} ---")
                print(f"텍스트: {text[:80]}...")
                print(f"참조: {item['keywords'][:3]}")
                print(f"KeyBERT: {keybert_kw[:3]}")
                print(f"TF-IDF: {tfidf_kw[:3]}")
                print(f"TextRank: {textrank_kw[:3]}")
                print(f"Hybrid: {hybrid_kw[:3]}")
        
        # 결과 저장
        self.predictions = {
            'keybert': keybert_predictions,
            'tfidf': tfidf_predictions,
            'textrank': textrank_predictions,
            'hybrid': hybrid_predictions
        }
        
        print("✅ 키워드 추출 완료!")
    
    def calculate_precision_recall_f1(self, predicted, true, k=5):
        """Precision@K, Recall@K, F1@K 계산"""
        pred_k = set([kw.lower().strip() for kw in predicted[:k] if kw])
        true_set = set([kw.lower().strip() for kw in true if kw])
        
        if len(true_set) == 0 or len(pred_k) == 0:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        intersection = pred_k.intersection(true_set)
        
        precision = len(intersection) / len(pred_k) if len(pred_k) > 0 else 0
        recall = len(intersection) / len(true_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def calculate_partial_match_score(self, predicted, true, k=5):
        """부분 매칭 점수 계산"""
        pred_k = [kw.lower().strip() for kw in predicted[:k] if kw]
        true_list = [kw.lower().strip() for kw in true if kw]
        
        if len(true_list) == 0 or len(pred_k) == 0:
            return 0
        
        matches = 0
        for pred in pred_k:
            for true_kw in true_list:
                if (pred == true_kw or 
                    pred in true_kw or 
                    true_kw in pred or
                    self._fuzzy_match(pred, true_kw)):
                    matches += 1
                    break
        
        return matches / len(pred_k) if len(pred_k) > 0 else 0
    
    def _fuzzy_match(self, str1, str2, threshold=0.7):
        """유사 매칭"""
        if len(str1) < 2 or len(str2) < 2:
            return False
        
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return (intersection / union) > threshold if union > 0 else False
    
    def calculate_semantic_similarity(self, predicted, true, k=5):
        """의미적 유사도 계산"""
        pred_k = [kw for kw in predicted[:k] if kw.strip()]
        true_list = [kw for kw in true if kw.strip()]
        
        if len(true_list) == 0 or len(pred_k) == 0:
            return 0
        
        try:
            pred_embeddings = self.semantic_model.encode(pred_k)
            true_embeddings = self.semantic_model.encode(true_list)
            
            similarities = cosine_similarity(pred_embeddings, true_embeddings)
            max_similarities = np.max(similarities, axis=1)
            
            return np.mean(max_similarities)
            
        except:
            return 0
    
    def calculate_all_metrics(self):
        """모든 평가 지표 계산"""
        print("\n📊 평가 지표 계산 중...")
        
        methods = ['keybert', 'tfidf', 'textrank', 'hybrid']
        metrics = ['precision', 'recall', 'f1', 'partial_match', 'semantic_sim']
        
        self.results = {}
        
        for method in methods:
            print(f"  - {method.upper()} 평가 중...")
            
            self.results[method] = {metric: [] for metric in metrics}
            predictions = self.predictions[method]
            
            for pred, true in zip(predictions, self.references):
                # 기본 지표
                basic_metrics = self.calculate_precision_recall_f1(pred, true, k=5)
                self.results[method]['precision'].append(basic_metrics['precision'])
                self.results[method]['recall'].append(basic_metrics['recall'])
                self.results[method]['f1'].append(basic_metrics['f1'])
                
                # 부분 매칭
                partial_score = self.calculate_partial_match_score(pred, true, k=5)
                self.results[method]['partial_match'].append(partial_score)
                
                # 의미적 유사도
                sem_sim = self.calculate_semantic_similarity(pred, true, k=5)
                self.results[method]['semantic_sim'].append(sem_sim)
        
        print("✅ 모든 평가 지표 계산 완료!")
    
    def create_visualizations(self):
        """결과 시각화"""
        print("\n🎨 시각화 생성 중...")
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        methods = ['keybert', 'tfidf', 'textrank', 'hybrid']
        metrics = ['precision', 'recall', 'f1', 'partial_match', 'semantic_sim']
        
        # 1. 박스플롯
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Korean Keyword Extraction Performance Distribution', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            data_to_plot = [self.results[method][metric] for method in methods]
            
            axes[row, col].boxplot(data_to_plot, labels=['KeyBERT', 'TF-IDF', 'TextRank', 'Hybrid'])
            axes[row, col].set_title(f'{metric.upper()}@5')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        # 빈 subplot 제거
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('korean_keyword_extraction_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 평균 성능 비교
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(metrics))
        width = 0.2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, method in enumerate(methods):
            means = [np.mean(self.results[method][metric]) for metric in metrics]
            stds = [np.std(self.results[method][metric]) for metric in metrics]
            
            ax.bar(x + i*width, means, width, yerr=stds, 
                   label=method.upper(), alpha=0.8, capsize=5, color=colors[i])
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Scores')
        ax.set_title('Korean Keyword Extraction Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('korean_keyword_extraction_barplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 히트맵
        heatmap_data = []
        for method in methods:
            row = [np.mean(self.results[method][metric]) for metric in metrics]
            heatmap_data.append(row)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data,
                   xticklabels=[m.upper() for m in metrics],
                   yticklabels=[m.upper() for m in methods],
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax)
        ax.set_title('Korean Keyword Extraction Performance Heatmap')
        plt.tight_layout()
        plt.savefig('korean_keyword_extraction_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 시각화 완료!")
    
    def save_results(self):
        """결과 저장"""
        print("\n💾 결과 저장 중...")
        
        # 상세 결과 CSV
        detailed_results = []
        for i in range(len(self.references)):
            row = {
                'sample_id': i,
                'text': self.test_texts[i][:200] + "..." if len(self.test_texts[i]) > 200 else self.test_texts[i],
                'true_keywords': '; '.join(self.references[i][:10])
            }
            
            for method in ['keybert', 'tfidf', 'textrank', 'hybrid']:
                row[f'{method}_keywords'] = '; '.join(self.predictions[method][i][:5])
                
                for metric in ['precision', 'recall', 'f1', 'partial_match', 'semantic_sim']:
                    row[f'{method}_{metric}'] = self.results[method][metric][i]
            
            detailed_results.append(row)
        
        df = pd.DataFrame(detailed_results)
        df.to_csv('korean_keyword_extraction_results.csv', index=False, encoding='utf-8-sig')
        
        # 요약 결과 JSON
        summary_data = {
            'experiment_info': {
                'num_samples': self.num_samples,
                'random_seed': self.random_seed,
                'evaluation_date': datetime.now().isoformat(),
                'dataset': 'KoreanSummarizeAiHub',
                'language': 'Korean',
                'methods': ['KeyBERT', 'TF-IDF', 'TextRank', 'Hybrid']
            },
            'performance_summary': {},
            'sample_predictions': {
                'keybert': self.predictions['keybert'][:3],
                'tfidf': self.predictions['tfidf'][:3],
                'textrank': self.predictions['textrank'][:3],
                'hybrid': self.predictions['hybrid'][:3],
                'references': self.references[:3]
            }
        }
        
        # 성능 요약 계산
        for method in ['keybert', 'tfidf', 'textrank', 'hybrid']:
            summary_data['performance_summary'][method] = {}
            for metric in ['precision', 'recall', 'f1', 'partial_match', 'semantic_sim']:
                scores = self.results[method][metric]
                summary_data['performance_summary'][method][metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'median': float(np.median(scores))
                }
        
        with open('korean_keyword_extraction_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print("✅ 결과 저장 완료!")
        print("  - korean_keyword_extraction_results.csv: 상세 결과")
        print("  - korean_keyword_extraction_summary.json: 통계 요약")
    
    def generate_report(self):
        """종합 보고서 생성"""
        print("\n📋 종합 보고서 생성 중...")
        
        methods = ['keybert', 'tfidf', 'textrank', 'hybrid']
        metrics = ['precision', 'recall', 'f1', 'partial_match', 'semantic_sim']
        
        report = []
        report.append("=" * 80)
        report.append("한국어 키워드 추출 모델 비교 평가 보고서")
        report.append("데이터셋: KoreanSummarizeAiHub")
        report.append("=" * 80)
        report.append(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"평가 샘플 수: {self.num_samples}")
        report.append(f"언어: 한국어")
        report.append("")
        
        # 성능 요약
        report.append("📊 성능 요약 (평균 ± 표준편차)")
        report.append("-" * 60)
        
        for metric in metrics:
            report.append(f"\n{metric.upper()}:")
            for method in methods:
                mean_score = np.mean(self.results[method][metric])
                std_score = np.std(self.results[method][metric])
                report.append(f"  {method.upper():>10}: {mean_score:.4f} ± {std_score:.4f}")
        
        report.append("")
        
        # 최고 성능 방법
        report.append("🏆 최고 성능 방법")
        report.append("-" * 40)
        
        for metric in metrics:
            best_method = max(methods, key=lambda m: np.mean(self.results[m][metric]))
            best_score = np.mean(self.results[best_method][metric])
            report.append(f"{metric.upper():>15}: {best_method.upper()} ({best_score:.4f})")
        
        report.append("")
        
        # 전체 승수 계산
        wins = {method: 0 for method in methods}
        for metric in metrics:
            best_method = max(methods, key=lambda m: np.mean(self.results[m][metric]))
            wins[best_method] += 1
        
        best_overall = max(wins, key=wins.get)
        report.append("🎯 종합 결론")
        report.append("-" * 40)
        report.append(f"• 전반적 최고 성능: {best_overall.upper()} ({wins[best_overall]}/{len(metrics)} 지표에서 1위)")
        
        # 실용적 권장사항
        report.append("")
        report.append("💡 실용적 권장사항")
        report.append("-" * 40)
        report.append("• 한국어 환경에서는 언어 특화 모델의 성능이 중요")
        report.append("• 요약문에서 키워드 추출 시 의미적 유사도가 핵심 지표")
        report.append("• 하이브리드 방법으로 안정적인 성능 확보 가능")
        report.append("• 공공기관 문서에서는 도메인 특화 후처리 필요")
        
        report.append("")
        report.append("=" * 80)
        
        # 보고서 저장 및 출력
        with open('korean_keyword_extraction_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        for line in report:
            print(line)
        
        print("\n✅ 보고서 생성 완료! (korean_keyword_extraction_report.txt)")
    
    def run_korean_evaluation(self):
        """한국어 키워드 추출 평가 전체 프로세스 실행"""
        print("🚀 한국어 키워드 추출 모델 비교 평가 시작!")
        print("=" * 60)
        
        try:
            # 1. 모델 로드
            self.load_models()
            
            # 2. 한국어 데이터셋 로드
            self.load_korean_dataset()
            
            if not self.test_data:
                print("❌ 테스트 데이터가 없습니다.")
                return
            
            # 3. 한국어 TF-IDF 준비
            self.prepare_korean_tfidf()
            
            # 4. 키워드 추출
            self.extract_all_keywords()
            
            # 5. 평가 지표 계산
            self.calculate_all_metrics()
            
            # 6. 시각화
            self.create_visualizations()
            
            # 7. 결과 저장
            self.save_results()
            
            # 8. 보고서 생성
            self.generate_report()
            
            print("\n🎉 한국어 키워드 추출 평가 완료!")
            print("생성된 파일들:")
            print("  - korean_keyword_extraction_results.csv")
            print("  - korean_keyword_extraction_summary.json")
            print("  - korean_keyword_extraction_report.txt")
            print("  - korean_keyword_extraction_*.png")
            
        except Exception as e:
            print(f"❌ 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

evaluator = KoreanKeywordExtractorComparison(num_samples=50, random_seed=42)
evaluator.run_korean_evaluation()