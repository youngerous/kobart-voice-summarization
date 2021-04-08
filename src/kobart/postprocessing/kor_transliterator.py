"""
* TODO : 보다 더 많은 학습시간, 데이터를 활용하여 다음과 같은 음역과정 없이 음성합성 모델이 영어나 숫자 읽는 법을 학습하도록 해야함.
        현재는 본 프로젝트가 중점적으로 다루고 있는 '뉴스 기사 요약문'을 커버할 정도의 기본적인 음역 모듈을 구축

* 본 모듈을 톻해 음역 가능한 있는 주요 단어 
        1) 기호 : 뉴스 기사에서 자주 쓰이는 날짜, 시각, %, %p,℃ 등
        2) 영어 : 1글자 및 3글자 이상의 영어 단어는 대체로 음역이 잘 되나, 두글자의 경우 불완전한 경우를 보임 
                 (e.g. 'yg': 이지(X), 'sm': 에스엠 (O))
        3) 숫자 : '억'단위의 자리수 및 소수점까지 한자어 수사(e.g. 일, 이, 삼...,)로 음역 가능

* 미해결된 음역 모호성 :
     * '-'기호 (e.g. '-10' : 마이너스십 '7-10' : 칠 에서 십)
     * 고유어/한자어 수사 구분 (e.g 2번쨰 : 두 번째, 2번지 : 이번지)

"""

import re
from engkor_transliterator import seq2seq_att_pt


####################
# 숫자 한글 음역  모델 #
####################
# surce :https://github.com/gritmind/engkor_transliterator

eng2kor_model = seq2seq_att_pt.Transliterator()
eng2kor_model.use_pretrained_model() # use pre-trained model

####################
# 숫자 한글 음역  함수 #
####################
# source : https://m.blog.naver.com/PostView.nhn?blogId=chandong83&logNo=221144077279&proxyReferer=https:%2F%2Fwww.google.com%2F

# 만 단위 자릿수
tenThousandPos = 4
# 억 단위 자릿수
hundredMillionPos = 9
txtDigit = ['', '십', '백', '천', '만', '억']
txtNumber = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
txtPoint = '쩜 '

def digit2txt(strNum):
    resultStr = ''
    digitCount = 0
    # print(strNum)
    #자릿수 카운트
    for ch in strNum:
        # ',' 무시
        if ch == ',':
            continue
        #소숫점 까지
        elif ch == '.':
            break
        digitCount = digitCount + 1


    digitCount = digitCount-1
    index = 0

    while True:
        notShowDigit = False
        ch = strNum[index]
        #print(str(index) + ' ' + ch + ' ' +str(digitCount))
        # ',' 무시
        if ch == ',':
            index = index + 1
            if index >= len(strNum):
                break;
            continue

        if ch == '.':
            resultStr = resultStr + txtPoint
        else:
            # 자릿수가 2자리이고 1이면 '일'은 표시 안함.
            # 단 '만' '억'에서는 표시 함
            if(digitCount > 1) and (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos) and int(ch) == 1:
                resultStr = resultStr + ''
            elif int(ch) == 0:
                resultStr = resultStr + ''
                # 단 '만' '억'에서는 표시 함
                if (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos):
                    notShowDigit = True
            else:
                resultStr = resultStr + txtNumber[int(ch)]


        # 1억 이상
        if digitCount > hundredMillionPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-hundredMillionPos]
        # 1만 이상
        elif digitCount > tenThousandPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-tenThousandPos]
        else:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount]

        if digitCount <= 0:
            digitCount = 0
        else:
            digitCount = digitCount - 1
        index = index + 1
        if index >= len(strNum):
            break;
        
        # DIFF(subin): 일십 -> 십 으로 출력되도록
        if len(resultStr) > 1 and '쩜' not in resultStr:
            if resultStr.startswith('일'):
                resultStr = resultStr[1:]
    return resultStr


class Transliterator():
    def __init__(self, text):
        self.text = text
        
        
        # TODO(subin): handle more heuristic cases.
        self.special_cases = {'6월': '유월', '10월': '시월', '코로나19':'코로나일구', '1시':'한시', '2시':'두시', '3시':'세시', '4시':'네시', '5시':'다섯시',
                             '6시': '여섯시', '7시':'일곱시', '8시':'여덟시', '9시':'아홉시', '10시': '열시', '11시':'열한시', '12시':'열두시'}
        # careful of ordering units
        self.units = {'m':'미터', 'g':'그램', '%':'퍼센트', '%p':'퍼센트포인트',
                      'mm':'밀리미터', 'cm':'센티미터','km':'킬로미터', 'kg':'킬로그램', '℃':'도', 'm/s':'미터퍼세크', 'km/h':'킬로미터퍼아우어'}
        
        
    def _heuristic_transliteration(self):
        
        for case, trans in self.special_cases.items():
             self.text = re.sub(case, trans, self.text)

        idx_list = []
        # The numerical value always precedes the unit
        for unit, trans in self.units.items():
            pattern = re.compile('[0-9]+\s*{}'.format(unit))
            groups = pattern.finditer(self.text)
          
            for group in groups:
                idx_list.append((group.start(), group.end(), unit)) 
                
        
        # reverse order to avoid index collision after replacing text.
        idx_list.reverse()
        for start, end, unit in idx_list:
            text1 = self.text[:start]
            text2 = self.text[start:]
            text2 = re.sub(unit, self.units[unit], text2)
            self.text = text1 + text2

        
    def _eng2kor(self):
        
        eng_idx_list = []
        pattern = re.compile('[a-zA-Z]+') 
        groups = pattern.finditer(self.text)
                
        for group in groups:
            eng_idx_list.append((group.start(), group.end()))
        
        # reverse order to avoid index collision after replacing text.
        eng_idx_list.reverse()
        for start, end in eng_idx_list:
            text1 = self.text[:start]
            text2 = self.text[start:]
            text2 = re.sub(text2[0:end-start], eng2kor_model.decode_sequence(text2[0:end-start])[0] , text2)
            self.text = text1 + text2
    

    def _num2kor(self):
        
        num_idx_list = []
        pattern = re.compile('[0-9]+\.?\d*') 
        groups = pattern.finditer(self.text)
        
        for group in groups:
            num_idx_list.append((group.start(), group.end()))
            
        # reverse order to avoid index collision after replacing text.
        num_idx_list.reverse()
        for start, end in num_idx_list:
            text1 = self.text[:start]
            text2 = self.text[start:]
            text2 = re.sub(text2[0:end-start], digit2txt(text2[0:end-start]), text2)
            self.text = text1 + text2
    
    
    def transliterate(self):
        self._heuristic_transliteration()
        self._eng2kor()
        self._num2kor()
        return self.text