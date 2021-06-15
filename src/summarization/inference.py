import argparse
from collections import OrderedDict
from pprint import pprint

import torch
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
try:
    from .distilbart import DistilBART
except:
    from distilbart import DistilBART
from transformers import BartForConditionalGeneration

MODEL = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
MODEL.to("cpu")
TOK = get_kobart_tokenizer()
INITIALIZED_WEIGHT = False
INITIALIZED_MODEL = False


def initialize(ckpt: str) -> None:
    global MODEL
    global INITIALIZED_WEIGHT
    # cpu load
    state_dict = torch.load(ckpt, map_location=torch.device("cpu"))

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if "module." in k else k  # remove `module.`
        new_state_dict[name] = v

    MODEL.load_state_dict(new_state_dict, strict=True)
    INITIALIZED_WEIGHT = True


@torch.no_grad()
def get_summarized_text(ckpt: str, text: str, n_enc: int = 6, n_dec: int = 6) -> str:
    global MODEL
    global INITIALIZED_MODEL
    global INITIALIZED_WEIGHT

    distilled = (n_enc != 6) or (n_dec != 6)
    if (not INITIALIZED_MODEL) and (distilled):
        MODEL = DistilBART(MODEL, n_enc=n_enc, n_dec=n_dec).to("cpu")
        INITIALIZED_MODEL = True
    if not INITIALIZED_WEIGHT:
        initialize(ckpt)

    MODEL.eval()
    text = text.replace("\n", "")
    input_ids = TOK.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    if distilled:
        output = MODEL.student.generate(
            input_ids, eos_token_id=1, max_length=512, num_beams=5
        )
    else:
        output = MODEL.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = TOK.decode(output[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, help="Checkpoint path to inference", default=None
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Korean text to summarize"
    )
    parser.add_argument("--n-enc", type=int, default=6, help="Number of encoder layer")
    parser.add_argument("--n-dec", type=int, default=6, help="Number of decoder layer")
    args = parser.parse_args()

    assert args.ckpt, "You should specify checkpoint path."
    assert args.text, "You should specify text to summarize."

    # original_text = """
    # 코로나19(신종 코로나 바이러스 감염증) 3차 대유행의 확산세가 거세지고 있다. 16일 신규 확진자 수는 또다시 1000명 선을 넘었다. 지난 13일 1030명 이후 사흘만에 1000명대로 올라선 것이다. 지난 1월20일 국내 첫 코로나19 확진자 발생 이후 최다 기록도 경신됐다.
    # 신규 확진자가 연일 증가하자 최근 한 주간 일평균 지역발생 확진자 수는 처음으로 `사회적 거리두기 3단계 범위`에 들어왔다. 3단계 범위는 전국 800~1000명 이상 또는 더블링 등 급격한 증가시에 해당된다.
    # 정부는 단기간에 확진자 수가 하루 1200명대까지 늘어날 수도 있을 전망하고, 거리두기 3단계 격상을 포함한 다양한 대책을 마련 중이다.
    # 중앙방역대책본부는 이날 0시 기준 코로나19 신규 확진자가 1078명 늘어 누적 4만5442명임을 전했다. 지난 15일(880)명보다 198명 증가한 수치다. 신규 확진자의 감염경로를 보면 지역발생이 1천54명, 해외유입이 24명이다. 지역발생 확진자는 지난 15일(848명)보다 206명 늘었다.
    # 확진자가 나온 지역을 보면 서울 373명, 경기 320명, 인천 64명 등 수도권이 757명이다. 수도권 이외 지역에서는 전북이 75명으로 가장 많고 이어 부산 41명, 충남 35명, 경북 28명, 대구 27명, 충북 22명, 경남 19명, 대전·제주 각 15명, 강원 8명, 울산 6명, 전남 4명, 광주 2명이다.
    # 사망자는 지난 15일보다 12명이 증가해 누적 612명으로 집계됐다. 국내 평균 치명률은 1.35%다. 상태가 악화한 위중증 환자는 지난 15일보다 21명 늘어난 226명이다.
    # """

    # original_text = """
    # SK텔레콤(대표 박정호)이 자체 개발한 데이터센터용 AI 반도체를 25일 공개했다.
    # 2024년 약 50조원 규모로 성장이 예상되는 AI 반도체 시장에 본격 진출한다는 계획이다.
    # AI 반도체는 인공지능 서비스의 구현에 필요한 대규모 연산을 초고속·저전력으로 실행하는 비메모리 반도체다.
    # 최근 데이터양이 기하급수적으로 늘면서 이를 처리하기 위한 AI 반도체의 필요성도 높아지고 있다.
    # 현재 이 시장은 엔비디아·인텔·구글 등 글로벌 빅테크 기업 중심으로 경쟁이 치열해지고 있다.
    # SK텔레콤은 이날 AI 반도체 ‘SAPEON X220’를 선보였다.
    # 기존 GPU(그래픽처리장치) 대비 딥러닝 연산 속도가 1.5배 빠르고, 데이터센터에 적용 시 데이터 처리 용량이 1.5배 증가한다.
    # 동시에 가격은 GPU의 절반 수준이고 전력 사용량도 80%에 불과하다.
    # SK텔레콤은 맞춤형 설계를 통해 ‘SAPEON X220’의 경쟁력을 확보했다고 강조했다.
    # 데이터 처리 역량 대부분을 동시다발적 데이터 처리에 활용하도록 설계해 효율성을 극대화했다는 설명이다.
    # 이 제품은 다양한 분야의 데이터센터에 즉시 적용 가능하다.
    # SK텔레콤은 국내외 다양한 사업자를 대상으로 AI 반도체 사업을 본격 추진할 계획이다.
    # """

    # original_text = """
    # 가수이자 화가로 전천후 활동 중인 '진한 음색' 솔비가 달콤커피 베란다라이브 무대를 통해 오랜만에 보컬리스트로서의 깊고 진한 매력을 선보인다.
    # 다날의 프랜차이즈 커피 전문 브랜드 달콤커피는 가수 솔비를 '이 달의 아티스트'로 선정하고, 달콤커피 매장에서 베란다라이브 공연을 실시한다고 5일 밝혔다.
    # 베란다라이브는 오는 15일과 22일 오후 8시부터 각각 달콤커피 논현점과 산본피트인점에서 열린다.
    # 신청 방법은 18일까지 달콤커피 공식 앱을 통해 응모 가능하며, 논현점 또는 산본피트인점에서 만원 이상 구매한 영수증 응모를 통해서도 참여할 수 있다.
    # 당첨자는 오는 11일(논현점), 19일(산본피트인점) 달콤커피 공식 페이스북을 통해 발표되며, 당첨자에게는 1인 당 2매의 티켓이 지급된다.
    # 11월의 아티스트로 선정된 솔비는 지난 2006년 혼성그룹 '타이푼'의 리드보컬로 데뷔, 그룹과 솔로 활동을 병행하며 진한 음색과 파워풀한 가창력으로 꾸준히 사랑받고 있는 실력파 보컬리스트다.
    # 최근 '음악하는 솔비'와 '미술하는 권지안(솔비의 본명)'의 협업, '셀프 콜라보레이션'이라는 독창적인 작업을 통해 음악과 미술의 경계를 넘어선 자신만의 장르를 개척하는 등 다재다능한 매력으로 폭넓은 지지를 받고 있다.
    # 솔비는 달콤커피 베란다라이브를 통해 특유의 파워풀한 가창력과 호소력 짙은 음색으로 오랜만에 '명품 보컬리스트'로서의 존재감을 유감없이 발휘할 예정이다.
    # 또 현장 이벤트를 통해 팬들과 소통하며, 특유의 상큼발랄하고 사랑스러운 매력을 선보이는 등 달콤한 시간을 함께할 계획이다.
    # 한편, '달콤커피 베란다라이브'는 매월 정동하, 김태우, 알리, 허각, 한동근, 플라이투더스카이, 홍대광, 하이어뮤직, 라비, 권진아 등 역량있는 뮤지션들을 '이달의 아티스트'로 선정하고, 전국의 달콤커피 매장에서 라이브 공연을 실시하는 달콤커피 만의 특별한 문화 이벤트다.
    # """

    # original_text = """
    # 주민들에게 서민금융 지원 제도를 알리고 맞춤형 서비스를 연계해 줄 협업체계가 구축된다.
    # 익산시는 4일 서민금융통합지원센터(센터장 이재갑)를 주축으로 근로복지공단 익산지사, 익산지역자활센터, 소상공인시장진흥공단 익산센터, 지역 소재 금융회사 등 유관기관과 간담회를 갖고‘서민금융 지역협의체’를 구축했다고 밝혔다.
    # 협약에 따라 각 기관은 앞으로 시민 개개인의 상황과 목적에 맞는 다양한 맞춤형 금융서비스를 안내하고 제공받을 수 있도록 지원한다.
    # 특히 소득기준 초과로 근로복지공단의 생계 자금지원이 어려운 근로자가 통합지원센터를 통해 지역 금융회사에서 대출을 받거나 긴급자금이 필요한 기초수급자가 통합지원센터와 연계해 미소금융 대출과 재산형성 지원을 받을 수 있도록 연계할 예정이다.
    # 아울러 이 날 간담회에서는 서민금융지원 강화를 위해 향후 운영방안과 참여기관별 지원제도를 공유해 유기적인 안내가 가능하도록 종합홍보물 제작 등에 대해서도 논의됐다.
    # 익산시는“서민금융 지역협의체의 활성화에 주력하고 금융·고용·복지 기관들과 유기적으로 협업해 시민들이 몰라서 이용하지 못하는 일이 없도록 다양한 매체를 통해 지속적으로 시민 홍보에 힘쓰겠다”고 밝혔다.
    # 한편 지난 2017년 개소한 익산 서민금융통합지원센터(☏1397)에서는 창업·사업운영·생활안정 등 필요자금을 담보나 보증 없이 저금리 지원, 생계자금 대출 보증 지원, 저금리 전환대출 지원, 채무조정 등 수요자가 한 자리에서 종합상담, 심사 및 지원까지 제공받을 수 있도록 서민금융진흥원, 신용회복위원회, 한국자산관리공사, 미소금융이 참여하고 있다.", "또한 전산시스템도 구축해 서민을 위한 맞춤형 원스톱 금융·복지 상담서비스를 운영하고 있으며 6월말 현재 총1,780건(미소금융1,240건, 신용회복위원회채무조정240건, 자산관리공사 바꿔드림론 등 300건)의 서민 금융 서비스가 지원됐다.
    # """

    # original_text = """
    # 전남 완도와 곡성의 청년들 유출 문제가 심각한 것으로 나타났다.
    # 호남지방통계청은 25일 완도군·곡성군과 지난 1·2월 업무협약을 맺고 청년들의 경제적 자립과 공존을 위해 청년정책을 수립할 목적으로 행정자료를 활용한 ‘곡성·완도군 청년통계’를 공표했다.
    # 호남지방통계청에 따르면 지난해 완도군 전체 인구 대비 청년은 30.6%(1만5천754명)로 전국 44.8%, 전남 37.5%보다 낮은 것으로 조사됐다.
    # 완도군 청년의 인구 이동을 보면 지난해에 299명이 순유출됐으며, 2013년과 2015년을 제외하고는 2003년 이후 순유출세가 지속되는 것으로 집계됐다.
    # 곡성군의 경우 지난해 전체 인구 대비 청년은 29.2%(8천657명)로 전국 44.8%, 전남 37.5%보다 낮았다.
    # 곡성군 청년의 인구이동을 보면 지난해에 393명이 순유출됐으며, 2013년을 제외하고는 2008년 이후 순유출세가 지속되는 것으로 나타났다.
    # 호남지방통계청과 완도군·곡성군청은 다양한 지표를 구성하는 ‘청년통계’를 활용해 취업, 창업 및 경제적 자립 등 청년들의 생활 만족도 향상과 지역발전에 기여할 것으로 보고 있다.
    # 이호석 호남지방통계청장은 “현장조사 위주의 통계조사 방식에서 공공데이터를 활용한 통계생산 방식으로의 체계 변화라고 할 ‘완도군·곡성군 청년통계’가 지역 청년 정책 자료로 적극적으로 활용돼 지역 발전에 기여하길 바란다”고 말했다.
    # """

    # args.text = original_text

    summary = get_summarized_text(args.ckpt, args.text, args.n_enc, args.n_dec)
    pprint(f"요약문: {summary}")
