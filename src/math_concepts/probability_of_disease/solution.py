def probability_of_disease(accuracy, prevalence):
    probability_healthy = 1 - prevalence
    probability_sick = prevalence

    inaccuracy = 1 - accuracy

    probability_of_testing_positive = probability_sick * accuracy + probability_healthy * inaccuracy
    probability_of_testing_negative = probability_sick * inaccuracy + probability_healthy * accuracy

    # probability that an individual has the disease given a positive test result
    probability_sick_given_positive = probability_sick * accuracy / probability_of_testing_positive

    # probability that an individual does not have the disease given a negative test result
    probability_healthy_given_negative = probability_healthy * accuracy / probability_of_testing_negative

    return [probability_sick_given_positive * 100, probability_healthy_given_negative * 100]
