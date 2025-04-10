import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    예측값과 실제값을 비교해서 멀티클래스 분류 정확도를 계산함
    preds: 예측 결과 (float 배열), shape: (N,)
    truths: 실제 라벨 (float/int 배열), shape: (N,)
    """
    # 소수점 반올림 후 비교해서 맞춘 개수 / 전체 개수 => 정확도 계산
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    """
    감정 분석 결과에 대해 weighted accuracy를 계산함
    예측값과 실제값이 0보다 큰지 여부를 기준으로 이진 분류 진행
    """
    true_label = (test_truth_emo > 0)       # 실제값이 양수인지 여부 (True/False)
    predicted_label = (test_preds_emo > 0)  # 예측값이 양수인지 여부 (True/False)

    # True Positive, True Negative 계산
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))

    # 전체 positive, negative 샘플 수
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    # weighted accuracy 계산: 정답 비율에 따라 가중 평균
    return (tp * (n / p) + tn) / (2 * n)


def eval_senti(results, truths, exclude_zero=False):
    """
    감정 분석 평가 지표들을 계산함 (MAE, 상관계수, F1-score 등)
    exclude_zero=True인 경우, 감정 중립(0)을 제외하고 평가함
    """
    # 결과 텐서를 numpy 배열로 변환
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    # 중립(0)을 제외할 경우 인덱스 필터링
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    # 감정 점수를 [-3, 3], [-2, 2] 범위로 클리핑해서 멀티클래스 평가
    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    # 평균 절대 오차 (MAE)
    mae = np.mean(np.absolute(test_preds - test_truth))

    # 예측과 실제의 상관계수
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    # 7클래스, 5클래스 멀티클래스 정확도
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # 이진 분류 F1-score (positive vs non-positive, 중립은 제외 가능)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

    # 정확도 계산
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc = accuracy_score(binary_truth, binary_preds)

    # 결과 출력
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", acc)

    return acc


def eval_food(results, truths):
    """
    음식 분류 (101개 클래스)에 대한 정확도, F1-score 계산
    results: softmax 출력 (N, 101)
    truths: 정답 라벨 (N,)
    """
    # 소프트맥스 결과에서 가장 확률 높은 클래스를 예측값으로 선택
    test_preds = results.view(-1, 101).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)  # 예측 클래스
    test_truth_i = test_truth                     # 정답 클래스

    # 가중 F1 점수, 정확도 계산
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)

    print("  - F1 Score: ", f1)
    print("  - Accuracy: ", acc)

    return acc


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    """
    Cosine Annealing 스케줄러 구현
    - base_value에서 final_value로 점진적으로 감소
    - warmup_epochs 동안은 start_warmup_value에서 base_value로 선형 증가

    :param base_value: 초기 학습률 (혹은 하이퍼파라미터 값)
    :param final_value: 최종 학습률
    :param epochs: 전체 에폭 수
    :param niter_per_ep: 에폭당 반복 수 (iteration 수)
    :param warmup_epochs: 워밍업 구간 에폭 수
    :param start_warmup_value: 워밍업 시작 값
    :return: 전체 학습 스케줄 배열
    """
    warmup_schedule = np.array([])

    # 워밍업 구간 스케줄 생성
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    # 나머지 구간은 코사인 곡선에 따라 값 감소
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    # 워밍업 + 메인 스케줄 합치기
    schedule = np.concatenate((warmup_schedule, schedule))

    # 길이 확인
    assert len(schedule) == epochs * niter_per_ep

    return schedule
